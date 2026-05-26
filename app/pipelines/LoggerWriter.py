from datetime import datetime

class LoggerWriter:
    def __init__(self, log_path):
        self.log_path = log_path
        self.buffer = ""

    def write(self, message):

        self.buffer += message

        while "\n" in self.buffer:

            line, self.buffer = self.buffer.split(
                "\n",
                1
            )

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(self.log_path, "a") as f:
                f.write(
                    f"[{timestamp}] {line}\n"
                )

    def flush(self):

        if self.buffer.strip():

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(self.log_path, "a") as f:
                f.write(
                    f"[{timestamp}] {self.buffer}\n"
                )

            self.buffer = ""