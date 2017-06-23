class Name(dict):
    def __contains__(self, item):
        return True


name = Name()
val = "a" in name
print(val)