s = "abcad"

max_ = 0

# if len(previous_string) > max_:
#         max_ = len(previous_string)


def find_max_sub_sequence(s):
    start = 0
    end = 1
    if len(s) == 0:
        return 1
    previous_string = s[0]
    while(end < len(s)):
        if s[end] in previous_string or end == len(s) - 1:
            break
        end += 1
        previous_string = s[start: end]

    if s[end] in previous_string:
        index = previous_string.index(s[end])
        rest_length = find_max_sub_sequence(s[index + 1:])
    else:
        previous_string = s[start:end + 1]
        rest_length = 0
    return (len(previous_string), rest_length)


print(find_max_sub_sequence(s))
