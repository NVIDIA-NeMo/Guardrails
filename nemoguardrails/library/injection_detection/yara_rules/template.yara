rule jinja_injection
{
    meta:
        author = "Erick Galinkin"
        description = "Detect possible server-side template injection attempts"
        date = "2025-04-02"

    strings:
        $template_open = "{{"
        $template_close = "}}"
        $open_condition = "{%"
        $close_condition = "%}"

    condition:
        ($template_open and
            $template_close and
            (@template_open < @template_close)
            ) or
        ($open_condition and
            $close_condition and
            (@open_condition < @close_condition)
            )
}
