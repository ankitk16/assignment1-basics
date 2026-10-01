test_string = "Hello 😄!"
utf8_encoded = test_string.encode("utf-16")  # returns bytes hello
print(utf8_encoded)
print(list(utf8_encoded))


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


b1 = 12 * 2**4
b2 = 0

bs = [b1, b2]

# ___Problem (unicode2____#
# Question-2
smile = "😄"
# decode_utf8_bytes_to_str_wrong(smile.encode("utf-8"))

# Question 3
b1 = 12 * 2**4
b2 = 0
bs = [b1, b2]
# decode_utf8_bytes_to_str_wrong(bs#)

# %%
