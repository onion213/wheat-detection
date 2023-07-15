import datetime


class Logger:
    def log(self, formattedMsg: str):
        print(self.timestamp, formattedMsg)

    def debug(self, msg: str):
        self.log(f"[DEBUG] {msg}")

    def info(self, msg: str):
        self.log(f"[INFO] {msg}")

    def warning(self, msg: str):
        self.log(f"[WARNING] {msg}")

    def error(self, msg: str):
        self.log(f"[ERROR] {msg}")

    def critical(self, msg: str):
        self.log(f"[CRITICAL] {msg}")

    @property
    def timestamp(self):
        return datetime.datetime.now().isoformat(timespec="seconds")
