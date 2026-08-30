"""Safely read bounded JSON or YAML declarations from local files."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat


def read_safe_declaration(path: Path, *, label: str) -> object:
    """Read a strict JSON-compatible declaration without path races."""

    directory_descriptors: list[int] = []
    directory_links: list[tuple[str, tuple[int, ...]]] = []
    file_descriptor = -1
    try:
        absolute = path.absolute()
        parts = absolute.parts
        if not absolute.is_absolute() or len(parts) < 2:
            raise ValueError(f"{label} 必须是安全的绝对 JSON 文件路径")

        def identity_axes(metadata: os.stat_result) -> tuple[int, ...]:
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )

        root_descriptor = os.open(
            absolute.anchor,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        directory_descriptors.append(root_descriptor)
        for component in parts[1:-1]:
            directory_descriptor = directory_descriptors[-1]
            before_component = os.stat(
                component, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if (
                stat.S_ISLNK(before_component.st_mode)
                or not stat.S_ISDIR(before_component.st_mode)
            ):
                raise ValueError(f"{label} JSON 路径不得包含符号链接")
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            directory_descriptors.append(next_descriptor)
            opened_component = os.fstat(next_descriptor)
            after_component = os.stat(
                component, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if not (
                identity_axes(before_component)
                == identity_axes(opened_component)
                == identity_axes(after_component)
            ):
                raise ValueError(f"{label} JSON 路径在读取时发生变化")
            directory_links.append((component, identity_axes(opened_component)))

        directory_descriptor = directory_descriptors[-1]
        leaf = parts[-1]
        metadata = os.stat(
            leaf, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ValueError(f"{label} 必须是普通且非符号链接的 JSON 文件")
        if metadata.st_size <= 0 or metadata.st_size > 4 * 1024 * 1024:
            raise ValueError(f"{label} JSON 文件大小超出安全范围")
        file_descriptor = os.open(
            leaf,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        before = os.fstat(file_descriptor)
        if identity_axes(metadata) != identity_axes(before):
            raise ValueError(f"{label} JSON 文件在读取前发生变化")
        payload = bytearray()
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError(f"{label} JSON 文件大小超出安全范围")
        after = os.fstat(file_descriptor)
        after_path = os.stat(
            leaf, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if not (
            identity_axes(before)
            == identity_axes(after)
            == identity_axes(after_path)
        ):
            raise ValueError(f"{label} JSON 文件在读取时发生变化")
        for index, (component, expected_identity) in enumerate(directory_links):
            parent_descriptor = directory_descriptors[index]
            child_descriptor = directory_descriptors[index + 1]
            linked_component = os.stat(
                component,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            opened_component = os.fstat(child_descriptor)
            if (
                stat.S_ISLNK(linked_component.st_mode)
                or not stat.S_ISDIR(linked_component.st_mode)
                or identity_axes(linked_component) != expected_identity
                or identity_axes(opened_component) != expected_identity
            ):
                raise ValueError(f"{label} JSON 路径在读取时发生变化")

        text = bytes(payload).decode("utf-8")
        if path.suffix.casefold() == ".json":

            def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for key, value in items:
                    if type(key) is not str:
                        raise ValueError(f"{label} JSON 字段名必须是字符串")
                    if key in result:
                        raise ValueError(f"{label} JSON 含有重复字段")
                    result[key] = value
                return result

            result = json.loads(
                text,
                object_pairs_hook=pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"{label} JSON 含有非有限数值 {token}")
                ),
            )
        elif path.suffix.casefold() in {".yaml", ".yml"}:
            import yaml  # type: ignore[import-untyped]

            class UniqueSafeLoader(yaml.SafeLoader):
                pass

            def unique_mapping(
                loader: UniqueSafeLoader, node: object, deep: bool = False
            ) -> dict[str, object]:
                if not isinstance(node, yaml.MappingNode):
                    raise ValueError(f"{label} YAML 映射无效")
                result: dict[str, object] = {}
                for key_node, value_node in node.value:
                    key = loader.construct_object(key_node, deep=deep)
                    if type(key) is not str:
                        raise ValueError(f"{label} YAML 字段名必须是字符串")
                    if key in result:
                        raise ValueError(f"{label} YAML 含有重复字段")
                    result[key] = loader.construct_object(value_node, deep=deep)
                return result

            UniqueSafeLoader.add_constructor(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                unique_mapping,
            )
            try:
                result = yaml.load(text, Loader=UniqueSafeLoader)
            except yaml.YAMLError as error:
                raise ValueError(f"{label} 不是有效的安全 YAML 声明") from error
        else:
            raise ValueError(f"{label} 声明必须使用 .json、.yaml 或 .yml")

        stack: list[tuple[object, int]] = [(result, 0)]
        seen_containers: set[int] = set()
        item_count = 0
        while stack:
            value, depth = stack.pop()
            item_count += 1
            if depth > 64 or item_count > 100_000:
                raise ValueError(f"{label} 声明结构超出安全范围")
            if type(value) is dict:
                identity = id(value)
                if identity in seen_containers:
                    raise ValueError(f"{label} 声明不得包含 YAML 别名或环")
                seen_containers.add(identity)
                if any(type(key) is not str for key in value):
                    raise ValueError(f"{label} 声明字段名必须是字符串")
                stack.extend((child, depth + 1) for child in value.values())
            elif type(value) is list:
                identity = id(value)
                if identity in seen_containers:
                    raise ValueError(f"{label} 声明不得包含 YAML 别名或环")
                seen_containers.add(identity)
                stack.extend((child, depth + 1) for child in value)
            elif type(value) is float:
                if not math.isfinite(value):
                    raise ValueError(f"{label} 声明不得包含非有限数值")
            elif value is not None and type(value) not in (str, int, bool):
                raise ValueError(f"{label} 声明只能包含严格 JSON 数据类型")
        return result
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} 不是可安全读取的 UTF-8 声明") from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)
