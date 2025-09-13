from typing import Any, Optional

from tilus.ir.builders import StmtBuilder
from tilus.ir.stmt import Stmt


class TilusContext:
    def bind_value(self) -> Optional[Any]:
        raise NotImplementedError()

    def post_process(self, body: Stmt) -> Stmt:
        raise NotImplementedError()


class ThreadGroupContext(TilusContext):
    def __init__(self, group_index: int, group_size: Optional[int], num_groups: Optional[int]):
        self.group_index: int = group_index
        self.num_groups: Optional[int] = num_groups
        self.group_size: Optional[int] = group_size

    def bind_value(self) -> None:
        return None

    def post_process(self, body: Stmt) -> Stmt:
        sb = StmtBuilder()

        with sb.thread_group(group_index=self.group_index, group_size=self.group_size, num_groups=self.num_groups):
            sb.append(body)
        return sb.flush_stmts()
