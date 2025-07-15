rule markdown_xss
{
    meta:
        author = "Erick Galinkin"
        description = "Detect potential cross-site scripting in Markdown"
        date = "2025-03-27"

    strings:
        $html_link = "href"
        $js = "javascript"
        $re_script = /<script>[^\n]+?<\x2Fscript>/i
        $re_md_embed = /\s?!\[[^\n]+\]\([^\n]+\)/
        $re_md_js = /\[[^\n]+\]\(javascript[\^n]+\)/


    condition:
        any of ($re*) or (@html_link < @js)
}
