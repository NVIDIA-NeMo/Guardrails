rule alwaysfalse
{
    meta:
        author = "Erick Galinkin"
        description = "Test always returns false"
        date = "2025-03-25"
    condition:
        false
}

rule alwaystrue
{
    meta:
        author = "Erick Galinkin"
        description = "Test always returns true"
        date = "2025-03-25"
    condition:
        true
}

rule test_string
{
    meta:
        author = "Erick Galinkin"
        description = "Detect 'test string please ignore'"
        date = "2025-03-25"
    strings:
        $str = "test string please ignore"
    condition:
        $str
}
rule test_regex
{
    meta:
        author = "Erick Galinkin"
        description = "Detect the character 't' at the beginning of a line, then anything on that line until 'ignore' or 'don't'"
        date = "2025-03-25"
    strings:
        $re = /^[t].*?(ignore|don\x27t)/
    condition:
        $re
}
