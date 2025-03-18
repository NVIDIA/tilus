from typing import List, Union, Sequence, Optional
from hidet.utils.doc import Doc, NewLine, NewLineToken


def doc_join(seq: List[Doc | str], sep: Union[Doc, str]) -> Doc:
    doc = Doc()
    for i in range(len(seq)):
        if i != 0:
            doc += sep
        doc += seq[i]
    return doc


def doc_join_lines(
    seq: Sequence[Union[str, Doc]],
    left: Union[Doc, str],
    right: Union[Doc, str],
    indent: Optional[int] = None,
    line_end_sep: str = ",",
) -> Doc:
    doc = Doc()
    if indent is None:
        indent = 4
    if len(seq) == 0:
        doc += left + right
        return doc
    else:
        num_lines = len(seq)
        doc += left
        for i in range(num_lines):
            doc += (NewLine() + seq[i]).indent(indent)
            if i != num_lines - 1:
                doc += line_end_sep
        doc += NewLine() + right
        return doc


def doc_comment(doc: Doc, comment_string: str = "# ") -> Doc:
    docs = list(doc.docs)
    new_docs: List[Union[NewLineToken, str]] = []
    for i, token in enumerate(docs):
        if isinstance(token, NewLineToken):
            new_docs.append(NewLineToken())
            if token.indent > 0:
                new_docs.append(" " * token.indent)
        else:
            new_docs.append(token)
    docs = new_docs
    new_docs = []
    if docs and not isinstance(docs[0], NewLineToken):
        new_docs.append(comment_string)
    for token in docs:
        if isinstance(token, NewLineToken):
            new_docs.append(token)
            new_docs.append(comment_string)
        else:
            new_docs.append(token)
    docs = new_docs
    ret = Doc()
    ret.docs = docs
    return ret


def doc_strip_parentheses(doc: Doc, left_paren: str = "(", right_paren: str = ")") -> Doc:
    seq = doc.docs
    if len(seq) == 0:
        return doc
    if not isinstance(seq[0], str) or seq[0].startswith(left_paren):
        return doc
    if not isinstance(seq[-1], str) or seq[-1].endswith(right_paren):
        return doc
    seq = seq.copy()
    assert isinstance(seq[0], str) and isinstance(seq[-1], str)
    seq[0] = seq[0].removeprefix(left_paren)
    seq[-1] = seq[-1].removesuffix(right_paren)

    ret = Doc()
    ret.docs = seq
    return ret
