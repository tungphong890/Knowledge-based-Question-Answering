from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.Wrapper import QueryBadFormed
import requests
import re
import os

app = Flask(__name__)

bart_tokenizer = AutoTokenizer.from_pretrained("bart_model", local_files_only=True)
bart_model     = AutoModelForSeq2SeqLM.from_pretrained("bart_model", local_files_only=True)

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader(
    "User-Agent",
    "KBQA-Demo/1.0 (tungphongmd@gmail.com)"
)
chat_history = []

def resolve_label_to_qid(label: str, language: str = "en") -> str:
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": 5,
        "type": "item"
    }
    resp = requests.get(url, params=params)
    data = resp.json()

    for entry in data.get("search", []):
        if entry.get("label", "").lower() == label.lower():
            return entry["id"]

    if data.get("search"):
        return data["search"][0]["id"]

    return None

def resolve_relation_to_pid(label: str, language: str = "en") -> str:
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": 1,
        "type": "property"
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if "search" in data and data["search"]:
        return data["search"][0]["id"]
    return None

def ground_sparql(raw_sparql: str) -> str:
    labels = re.findall(r"\{([^}]+)\}", raw_sparql)
    grounded = raw_sparql
    for label in labels:
        qid = resolve_label_to_qid(label)
        if qid:
            grounded = grounded.replace(f"{{{label}}}", f"wd:{qid}")
    return grounded

def generate_sparql(question: str) -> str:
    tokenizer, model = bart_tokenizer, bart_model

    prompt = f"generate query: {question}"
    inputs = tokenizer(prompt, return_tensors='pt')
    outs = model.generate(**inputs, max_length=128)
    raw_query_body = tokenizer.decode(outs[0], skip_special_tokens=True).strip()
    raw_query_body = ground_sparql(raw_query_body)

    if "SELECT" in raw_query_body and "WHERE" in raw_query_body:
        return raw_query_body

    qid_match = re.search(r"Q\d+", raw_query_body)
    prop_match = re.search(r"What is the (.*?) of", question, re.IGNORECASE)
    pid = None
    if prop_match:
        pid = resolve_relation_to_pid(prop_match.group(1))

    if qid_match and pid:
        qid = qid_match.group(0)
        return f"""
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT DISTINCT ?answerLabel WHERE {{
  wd:{qid} wdt:{pid} ?answer .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT 5
"""

    if "WHERE" in raw_query_body:
        return f"""
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT DISTINCT ?answerLabel WHERE {{
  {raw_query_body}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT 1
"""

    if qid_match:
        qid = qid_match.group(0)
        return f"""
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>

SELECT DISTINCT ?answerLabel WHERE {{
  wd:{qid} ?prop ?answer .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT 5
"""

    return "SELECT ?answer WHERE { ?s ?p ?o } LIMIT 1"

def run_sparql(query: str):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        res = sparql.query().convert()
    except QueryBadFormed as e:
        return ['error'], [{'error': str(e), 'query': query}]
    except Exception as e:
        return ['error'], [{'error': f"Request failed: {e}"}]

    vars_ = res['head']['vars']
    rows = [{v: b[v]['value'] if v in b else '' for v in vars_} for b in res['results']['bindings']]
    return vars_, rows

def render_html_table(vars_, rows):
    header = ''.join(f'<th>{v}</th>' for v in vars_)
    body = ''.join('<tr>' + ''.join(f'<td>{r[v]}</td>' for v in vars_) + '</tr>' for r in rows)
    return f'<table border="1"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        q = request.form['question'].strip()
        sparql_q = generate_sparql(q)
        print("Model: BART")
        print("Generated SPARQL query:\n", sparql_q)
        vars_, rows = run_sparql(sparql_q)
        chat_history.append({
            'question': q,
            'vars': vars_,
            'rows': rows,
            'query': sparql_q
        })
    return render_template('index.html', history=chat_history, render_html_table=render_html_table)

if __name__ == '__main__':
    app.run(debug=True)