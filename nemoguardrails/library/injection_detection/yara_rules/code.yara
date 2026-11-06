rule import_shells
{
    meta:
        author = "Erick Galinkin"
        description = "Detect import of the os, cmd, subprocess, shutil modules"
        date = "2025-03-27"

    strings:
        $from = "from"
        $imp = "import"
        $import_token = /\b(__import__|import_module)\b/
        $re_import = /\bimport[ \t\r\n\\]+([A-Za-z_][A-Za-z0-9_.]*([ \t\r\n\\]+as[ \t\r\n\\]+[A-Za-z_][A-Za-z0-9_]*)?[ \t\r\n\\]*,[ \t\r\n\\]*)*(os|cmd|subprocess|shutil|posix)(\.[A-Za-z_][A-Za-z0-9_]*)*([ \t\r\n\\]+as[ \t\r\n\\]+[A-Za-z_][A-Za-z0-9_]*)?[ \t]*($|[,;\r\n#\x22\x27\x60\x29\x5D\x7D])/
        $re_from = /\bfrom[ \t\r\n\\]+(os|cmd|subprocess|shutil|posix)(\.[A-Za-z_][A-Za-z0-9_]*)*[ \t\r\n\\]+import\b/

        $mod_os = "os"
        $mod_cmd = "cmd"
        $mod_subprocess = "subprocess"
        $mod_shutil = "shutil"

    condition:
        // Preserve legacy matches; new statement matches require module targets.
        ($imp and
            any of ($mod*) and
            for any of ($mod*) : (@imp < @)
            ) or
        ($imp and
            $from and
            any of ($mod*) and
            for any of ($mod*) : (@from < @) and
            for any of ($mod*) : (@ < @imp)
            ) or
        any of ($re*) or
        ($import_token and
            any of ($mod*) and
            for any of ($mod*) : (@import_token < @[#])
            ) or
        ($import_token and
            $from and
            any of ($mod*) and
            for any of ($mod*) : (
                for any i in (1..#) : (@from < @[i] and @[i] < @import_token[#import_token])
            )
            )
}

rule import_networking
{
    meta:
        author = "Erick Galinkin"
        description = "Detect import of Python networking libraries"
        date = "2025-03-27"

    strings:
        $from = "from"
        $imp = "import"
        $import_token = /\b(__import__|import_module)\b/
        $re_import = /\bimport[ \t\r\n\\]+([A-Za-z_][A-Za-z0-9_.]*([ \t\r\n\\]+as[ \t\r\n\\]+[A-Za-z_][A-Za-z0-9_]*)?[ \t\r\n\\]*,[ \t\r\n\\]*)*(socket|_socket|asyncio|http|httpx|bs4|requests|mechanize|urllib|urllib3|asyncssh)(\.[A-Za-z_][A-Za-z0-9_]*)*([ \t\r\n\\]+as[ \t\r\n\\]+[A-Za-z_][A-Za-z0-9_]*)?[ \t]*($|[,;\r\n#\x22\x27\x60\x29\x5D\x7D])/
        $re_from = /\bfrom[ \t\r\n\\]+(socket|_socket|asyncio|http|httpx|bs4|requests|mechanize|urllib|urllib3|asyncssh)(\.[A-Za-z_][A-Za-z0-9_]*)*[ \t\r\n\\]+import\b/

        $mod_socket = "socket"
        $mod_asyncio = "asyncio"
        $mod_http = "http"
        $mod_soup = "bs4"
        $mod_requests = "requests"
        $mod_mechanize = "mechanize"
        $mod_urllib = "urllib"
        $mod_asyncssh = "asyncssh"

    condition:
        // Preserve legacy matches; new statement matches require module targets.
        ($imp and
            any of ($mod*) and
            for any of ($mod*) : (@imp < @)
            ) or
        ($imp and
            $from and
            any of ($mod*) and
            for any of ($mod*) : (@from < @) and
            for any of ($mod*) : (@ < @imp)
            ) or
        any of ($re*) or
        ($import_token and
            any of ($mod*) and
            for any of ($mod*) : (@import_token < @[#])
            ) or
        ($import_token and
            $from and
            any of ($mod*) and
            for any of ($mod*) : (
                for any i in (1..#) : (@from < @[i] and @[i] < @import_token[#import_token])
            )
            )
}
