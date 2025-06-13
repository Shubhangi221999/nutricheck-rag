from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

nutrition_knowledge = [
    "High fructose corn syrup is linked to obesity and diabetes.",
    "Added sugars should be limited according to WHO.",
    "Palm oil contains saturated fat and may affect heart health.",
    "Whole food ingredients like oats, almonds, and dates are healthier.",
    "Natural flavors are often synthetic and offer no nutritional value.",
    "Short ingredient lists usually mean less processed food.",
    "Preservatives like BHT and sodium benzoate are controversial.",
    "Artificial sweeteners may affect gut microbiome in some people.",
    "Fiber-rich foods promote gut health and satiety.",
    "Processed oils are generally less healthy than cold-pressed oils."
]

# Embed & Index
model = SentenceTransformer("all-MiniLM-L6-v2")
kb_embeddings = model.encode(nutrition_knowledge)
index = faiss.IndexFlatL2(kb_embeddings.shape[1])
index.add(np.array(kb_embeddings))

def retrieve_facts(text, k=4):
    q = model.encode([text])
    _, I = index.search(q, k)
    return [nutrition_knowledge[i] for i in I[0]]

# Load LLM
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto")
pipeline_llm = pipeline("text-generation", model=llm, tokenizer=tokenizer, max_new_tokens=512)

def compare_ingredients(ingr_a, ingr_b, dietary_pref=""):
    facts_a = retrieve_facts(ingr_a)
    facts_b = retrieve_facts(ingr_b)
    pref = f"The user prefers {dietary_pref} foods." if dietary_pref else ""
    
    prompt = f"""
{pref}
Compare the following two food products and recommend the healthier one. Justify your reasoning using scientific nutrition knowledge.

Product A Ingredients: {ingr_a}
Relevant Facts: {facts_a}

Product B Ingredients: {ingr_b}
Relevant Facts: {facts_b}

Output a detailed comparison and final recommendation.
"""
    return pipeline_llm(prompt)[0]['generated_text']
