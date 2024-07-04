import subprocess
import time
import sys
import os


if __name__ == "_main_":

    try:

        mpg123 = subprocess.check_output(
            "which mpg123", shell=True
        ).strip().decode("utf-8")
        notification_sound = os.path.join(
            os.path.dirname(os.path.abspath(_file_)), "notification.mp3"
        )

        with open(os.devnull, "w") as trashcan:

            notify = lambda: subprocess.call(
                [mpg123, "-q", notification_sound],
                stdout=trashcan,
                stderr=subprocess.STDOUT
            )

            # Here we go:
            notify()
            while True:
                time.sleep(60*20) # 20 minutes
                notify()
                for _ in range(20):
                    time.sleep(1)
                notify()

    except KeyboardInterrupt:
        sys.exit(0)