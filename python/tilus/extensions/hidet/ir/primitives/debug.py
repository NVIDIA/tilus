from typing import Optional

from hidet.ir.stmt import BlackBoxStmt, Stmt


def comment(comment_string: str, style: Optional[str] = None) -> Stmt:
    """
    Generate a comment statement.

    usage:
    > comment("This is a comment.")
    // This is a comment.

    > comment("This is a comment.\nThis is the second line.")
    /*
     * This is a comment.
     * This is the second line.
     */

    > comment("This is a comment.", style='//')
    // This is a comment.

    > comment("This is a comment.", style='/*')
    /*
     * This is a comment.
     */

    > comment("This is a comment.\nThis is the second line.", style='//')
    // This is a comment.
    // This is the second line.
    """
    lines = comment_string.split("\n")

    if style is None:
        if len(lines) > 1:
            style = "/*"
        else:
            style = "//"

    if style not in ["//", "/*"]:
        raise ValueError('Invalid style: "{}", candidates: "//", "/*".'.format(style))

    if style == "/*":
        content = "\n".join(["/*"] + [" * " + line for line in lines] + [" */"])
    else:
        content = "\n".join(["// " + line for line in lines])
    return BlackBoxStmt(template_string=content)
