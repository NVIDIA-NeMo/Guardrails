rule import_shells
{
    meta:
        author = "Erick Galinkin"
        description = "Detect import of the os, cmd, subprocess, shutil modules"
        date = "2025-03-27"

    strings:
        $from = "from"
        $imp = "import"

        $mod_os = "os"
        $mod_cmd = "cmd"
        $mod_subprocess = "subprocess"
        $mod_shutil = "shutil"

    condition:
        ($imp and
            any of ($mod*) and
            for any of ($mod*) : (@imp < @)
            ) or
        ($imp and
            $from and
            any of ($mod*) and
            for any of ($mod*) : (@from < @) and
            for any of ($mod*) : (@ < @imp)
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

        $mod_socket = "socket"
        $mod_asyncio = "asyncio"
        $mod_http = "http"
        $mod_soup = "bs4"
        $mod_requests = "requests"
        $mod_mechanize = "mechanize"
        $mod_urllib = "urllib"
        $mod_asyncssh = "asyncssh"

    condition:
        ($imp and
            any of ($mod*) and
            for any of ($mod*) : (@imp < @)
            ) or
        ($imp and
            $from and
            any of ($mod*) and
            for any of ($mod*) : (@from < @) and
            for any of ($mod*) : (@ < @imp)
            )
}
