from flask import Flask, render_template, request
from config import DATA_DIR, RANK, USE_IDF
from search_engine.search_matrix import load_search_matrix
from search_engine.search_engine import SearchEngine


app = Flask(__name__)

search_matrix = load_search_matrix(DATA_DIR, RANK, USE_IDF)
search_engine = SearchEngine(search_matrix)


@app.route("/", methods=["GET", "POST"])
def index():
    query = request.form["query"]
    noise_removal = "noise_removal" in request.form
    num_of_results = int(request.form["num_results"])

    results = search_engine.search(query, num_of_results, noise_removal)

    return render_template(
        "index.html",
        result=results,
        num_of_results=num_of_results,
        noise_removal=noise_removal,
    )


if __name__ == "__main__":
    app.run(debug=True)
