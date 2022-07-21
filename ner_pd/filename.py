class Filename:
    """
    A support class in terms of filenames handling
    """

    def __init__(self, relpath: str, name: str, extension: str):
        self.relpath = relpath
        self.name = name
        self.extension = extension

    def get(self) -> str:
        return "{}/{}{}".format(self.relpath, self.name, self.extension)

    def get_without_extension(self) -> str:
        return "{}/{}".format(self.relpath, self.name)

    def get_name(self) -> str:
        return self.name
