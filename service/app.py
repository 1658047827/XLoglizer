from flask import Flask

app = Flask("XLoglizer")


@app.route("/")
def hello():
    return "Hello Flask!"


if __name__ == "__main__":
    app.run()
