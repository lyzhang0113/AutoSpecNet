def format_significant(value, total_chars=4, sig_digits=4):
    import math
    if value == 0:
        return f"{0:.{sig_digits-1}f}".ljust(total_chars)
    else:
        # Determine the number of digits before the decimal
        whole_digits = math.floor(math.log10(abs(value))) + 1 if value != 0 else 1
        # Calculate the decimal places to maintain a fixed total length
        decimal_places = max(0, sig_digits - whole_digits)
        formatted = f"{value:.{decimal_places}f}"
        return formatted.ljust(total_chars)