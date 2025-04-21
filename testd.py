# test_python_run.py

import datetime

def say_hello(name):
    print(f"Hello, {name}!")
    print("Today's date is:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Run main
if __name__ == "__main__":
    say_hello("Bestpick User")
