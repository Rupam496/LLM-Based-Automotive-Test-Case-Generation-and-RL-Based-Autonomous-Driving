class V2XChannel:
    """
    Simple ideal C-V2X communication channel.

    This is a simulation abstraction, NOT a real C-V2X protocol stack.

    Assumptions for the first version:
        - no packet loss
        - no latency
        - no jitter
        - no stale messages

    Each vehicle sends its current state to the other vehicle.
    """

    def __init__(self):
        self.messages = {
            "A": None,
            "B": None,
        }

    def send(self, sender_id, receiver_id, message):
        """
        Send a message from one vehicle to another.

        With the ideal channel, the message is available
        to the receiver immediately.
        """
        if sender_id not in ("A", "B"):
            raise ValueError("sender_id must be 'A' or 'B'.")

        if receiver_id not in ("A", "B"):
            raise ValueError("receiver_id must be 'A' or 'B'.")

        if sender_id == receiver_id:
            raise ValueError("A vehicle cannot send to itself.")

        self.messages[receiver_id] = message.copy()

    def receive(self, receiver_id):
        """
        Return the most recently received message.

        Returns None if no message has been received yet.
        """
        if receiver_id not in ("A", "B"):
            raise ValueError("receiver_id must be 'A' or 'B'.")

        if self.messages[receiver_id] is None:
            return None

        return self.messages[receiver_id].copy()

    def reset(self):
        """Clear all messages at the beginning of an episode."""
        self.messages["A"] = None
        self.messages["B"] = None
