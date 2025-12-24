"""
Continuation
"""


class Continuation:
    """
    A class representing a continuation.
    """

    def __init__(self, status: str, details: str, timestamp: str):
        """
        Initialize a Continuation object.

        Args:
            status (str): The status of the continuation.
            details (str): The details of the continuation.
            timestamp (str): The timestamp of the continuation.
        """
        self.status = status
        self.details = details
        self.timestamp = timestamp

    def __str__(self) -> str:
        """
        Returns:
            str: A string representation of the Continuation object.
        """
        return f"Status: {self.status}, Details: {self.details}, Timestamp: {self.timestamp}"

    def __repr__(self) -> str:
        """
        Returns:
            str: A string representation of the Continuation object.
        """
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """
        Returns:
            bool: True if the Continuation objects are equal, False otherwise.
        """
        if isinstance(other, Continuation):
            return (
                self.status == other.status
                and self.details == other.details
                and self.timestamp == other.timestamp
            )
        return False

    def __ne__(self, other: object) -> bool:
        """
        Returns:
            bool: True if the Continuation objects are not equal, False otherwise.
        """
        return not self == other

    def __hash__(self) -> int:
        """
        Returns:
            int: A hash value for the Continuation object.
        """
        return hash((self.status, self.details, self.timestamp))
