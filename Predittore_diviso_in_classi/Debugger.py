from time import gmtime, strftime

class Debugger:

    PRINT_ENABLED = True
    LOG_FILE_PATH = "log.txt"

    def __init__(self):
        self.logs = []
        self.log_file = open(Debugger.LOG_FILE_PATH, 'a')

    def print(self, text):
        out = "\n\n- {}: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), text)
        self.logs.append(out)
        self.log_file.write(out)
        if Debugger.PRINT_ENABLED:
            print(text)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.log_file.close()

