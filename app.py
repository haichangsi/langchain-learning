from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from ice_breaker.ice_breaker import ice_breaker_main


load_dotenv("ice_breaker/.env")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    summary, profile_pic = ice_breaker_main("gpt", name)
    return jsonify(
        {
            "summary": summary.summary,
            "facts": summary.facts,
            "ice_breakers": summary.ice_breakers,
            "topics_of_interest": summary.topics_of_interest,
            "profile_pic": profile_pic,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
