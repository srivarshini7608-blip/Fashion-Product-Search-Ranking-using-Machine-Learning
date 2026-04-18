# 👗 Fashion Product Search Ranking using Machine Learning



## 📌 Project Overview

This project builds a **search ranking system for fashion products** using user behavior signals such as clicks, add-to-cart, purchases, and dwell time.

The system learns to rank products based on relevance using a machine learning ranking model.



## 🎯 Objective

To improve search results by ranking products according to user preferences and interactions.



## 🧠 Concept Used

We use **Learning-to-Rank**, where the model learns how to order items instead of predicting a single label.

---

# 🚀 Step-by-Step Implementation

---

## 🔹 Step 1: Install Required Libraries

```python id="s1"
!pip install lightgbm
```

---

## 🔹 Step 2: Import Libraries

```python id="s2"
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
```

---

## 🔹 Step 3: Create Dataset

We simulate fashion search data with user behavior.

```python id="s3"
np.random.seed(42)

queries = ["red dress", "black shoes", "blue jeans"]
products_per_query = 15

data = []

for qid, query in enumerate(queries):
    for pid in range(products_per_query):
        price = np.random.randint(500, 5000)
        rating = np.random.uniform(2, 5)
        
        click = np.random.choice([0,1], p=[0.5,0.5])
        add_to_cart = np.random.choice([0,1], p=[0.7,0.3]) if click else 0
        purchase = np.random.choice([0,1], p=[0.85,0.15]) if add_to_cart else 0
        
        dwell_time = np.random.randint(5, 300) if click else np.random.randint(1, 10)
        
        relevance = (
            0.3 * click +
            0.3 * add_to_cart +
            0.4 * purchase +
            0.001 * dwell_time
        )

        data.append([
            qid, query, f"product_{pid}",
            price, rating,
            click, add_to_cart, purchase,
            dwell_time, relevance
        ])

df = pd.DataFrame(data, columns=[
    "query_id", "query", "product_id",
    "price", "rating",
    "click", "add_to_cart", "purchase",
    "dwell_time", "relevance"
])

df.head()
```

---

## 🔹 Step 4: Feature Engineering

```python id="s4"
df["normalized_price"] = df["price"] / df["price"].max()
df["ctr"] = df.groupby("query")["click"].transform("mean")

features = [
    "price", "rating",
    "click", "add_to_cart", "purchase",
    "dwell_time", "normalized_price", "ctr"
]

X = df[features]
y = df["relevance"]

group = df.groupby("query_id").size().to_list()
```

---

## 🔹 Step 5: Train Ranking Model

```python id="s5"
model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=50
)

model.fit(X, y, group=group)
```

---

## 🔹 Step 6: Predict Scores

```python id="s6"
df["score"] = model.predict(X)
```

---

## 🔹 Step 7: Rank Products

```python id="s7"
ranked = df.sort_values(
    ["query", "score"],
    ascending=[True, False]
)

ranked.head(10)
```

---

## 🔹 Step 8: View Results for One Query

```python id="s8"
result = ranked[ranked["query"] == "red dress"]

result[["product_id", "price", "rating", "score"]].head(10)
```

---

## 🔹 Step 9: Visualize Ranking

```python id="s9"
top10 = result.head(10)

plt.figure(figsize=(8,5))
plt.barh(top10["product_id"], top10["score"])
plt.gca().invert_yaxis()
plt.title("Top Ranked Products for 'red dress'")
plt.show()
```

---

## 🔹 Step 10: Evaluate Model

We use NDCG (ranking metric):

```python id="s10"
true_rel = df["relevance"].values.reshape(1, -1)
pred_rel = df["score"].values.reshape(1, -1)

print("NDCG Score:", ndcg_score(true_rel, pred_rel))
```

---

# 📊 Output

* Products are ranked based on predicted relevance score
* Higher engagement products appear at the top
* Visualization shows ranking clearly

---

# 📷 Example Result

### Query: "red dress"

Top ranked products:

* High purchase probability
* High user engagement

---

# 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* LightGBM
* Matplotlib

---

# 💡 Applications

* E-commerce search systems
* Recommendation engines
* Personalized ranking

---

# 🚀 Future Improvements

* Use real datasets
* Add NLP-based query matching
* Build web interface
* Use deep learning models

---

# 🔗 Project Structure

```
fashion-search-ranking/
│
├── notebook.ipynb
├── README.md
└── report.pdf
```

---
