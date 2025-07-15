rule sql_injection
{
    meta:
        author = "Erick Galinkin"
        description = "Detect possible SQL injection attempts"
        date = "2025-03-27"

    strings:
        $method_select = "SELECT" nocase
        $method_alter = "ALTER" nocase
        $method_add = "ADD" nocase
        $method_create = "CREATE" nocase
        $method_drop = "DROP" nocase
        $method_exec = "EXEC" nocase
        $method_union = "UNION" nocase
        $method_insert = "INSERT" nocase
        $method_upsert = "UPSERT" nocase
        $method_delete = "DELETE" nocase
        $method_truncate = "TRUNCATE" nocase

        $re_dash_comment = /--[^\r\n]+?/i
        $re_slashstar_comment = /\x2F\*[^\r\n\*\x2f]+?$/i
        $re_single_quote = /^([^']*'([^']*'[^']*')*[^']*')?[^']*'[^']+$/i
        $re_semicolon = /;[^\r\n]+?/i
        $re_char = /(cha?r\(\d+\)([,+]|\|\|)?)+/i
        $re_system_catalog = /(SELECT|FROM)\s*?pg_\w+?/i

    condition:
        any of ($method*) and any of ($re*)
}
