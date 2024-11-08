class DataNotFoundError(Exception):
    pass


class DataExistsError(Exception):
    """Tried to insert a value that already exists exactly once"""

    def __init__(self, idx: tuple, table: str):
        """pass the (id,) of the existing entry and the table
        we pass this as a tuple because we usually get these results by
        a fetch method which returns a tuple"""
        super().__init__()
        self.idx = idx[0]
        self.table = table


class DBConsistencyError(Exception):
    """Found multiple existing values, this should never happen and is critical"""

    pass
