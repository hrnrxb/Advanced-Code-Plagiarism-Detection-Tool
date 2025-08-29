# 🔍 Advanced Code Plagiarism Detection Tool

Tired of manually checking student assignments for copying?  
This **Python-based Plagiarism Detector** uses advanced code analysis techniques to automatically identify suspicious similarities between student submissions. It goes beyond simple text matching — it understands code *structure*, making it nearly impossible to fool with basic tricks like renaming variables or reformatting.

Perfect for teachers, TAs, or coding bootcamps looking to **maintain academic integrity** with minimal effort.

---

## ✨ Key Features: Why This Tool Stands Out

* **Multi-Method Detection Engine:**
  Combines **5 different analysis techniques** for maximum accuracy:
  - 🌲 **AST (Abstract Syntax Tree) Analysis** – Understands real code structure
  - 🔢 **Jaccard & Cosine Similarity** – Quantifies structural overlap
  - 🧠 **TF-IDF + Machine Learning** – Smart text pattern recognition
  - 🔤 **Sequence Matching** – Catches character-level copying
  - 🏗️ **Structural Fingerprinting** – Ignores variable names, focuses on logic

* **Smart Normalization:**
  Automatically converts all variable and function names to `var_0`, `func_1`, etc.  
  This means **renaming variables won’t fool the system**!

* **Visual Network Graphs:**
  See clusters of students with suspiciously similar code — perfect for identifying group copying.

* **Comprehensive Dashboard:**
  6 detailed charts showing score distributions, method correlations, and top suspicious pairs.

* **Clear Risk Levels:**
  Instantly see which cases need attention:
  - 🔴 **Very High Risk** (80%+) – Almost certain plagiarism
  - 🟠 **High Risk** (70–79%) – Strong evidence of copying
  - 🟡 **Moderate Risk** (60–69%) – Worth reviewing
  - 🟢 **Low Risk** (50–59%) – Minor similarities

* **CSV Export:**
  All results are saved to `advanced_plagiarism_results.csv` for documentation and review.

---

## 🚀 How It Works: Simple 1–100% Explanation

### 📥 Step 1: Load Student Code (10%)
- Scans the `homeworks/` folder
- Reads every `.py` file
- Stores student names and their code

### 🧱 Step 2: Build Code Blueprints (20%)
- Converts each Python file into an **Abstract Syntax Tree (AST)**
- Think of this as turning code into a LEGO model — same structure, different colors

### 🔍 Step 3: Normalize & Fingerprint (30%)
- Renames all variables/functions to generic labels (`var_0`, `func_1`)
- Counts key elements: loops, conditionals, function calls, etc.
- Creates a unique "fingerprint" for each submission

### 🕵️ Step 4: Run 5 Detection Methods (50%)
Each pair of students is analyzed using:

| Method | What It Catches |
|-------|-----------------|
| **Jaccard** | Direct copy-paste |
| **Cosine** | Slight modifications |
| **Structural** | Same logic, different names |
| **TF-IDF** | ML-powered text similarity |
| **Sequence** | Character-by-character copying |

### 📊 Step 5: Combine & Score (70%)
- Weighted average creates a **Combined Score**
- Risk level assigned based on threshold
- Results sorted from most to least suspicious

### 🖼️ Step 6: Generate Visual Reports (90%)

### 📈 1. Network Graph
- Students = nodes
- Suspicious pairs = edges
- Node color = number of connections
- Edge thickness = similarity score

### 📊 2. Statistics Dashboard
Six insightful plots:
1. Combined score distribution
2. Method vs. combined score scatter
3. Correlation heatmap
4. Risk level pie chart
5. Box plots for all methods
6. Top 10 most suspicious pairs

### 📤 Step 7: Final Output (100%)
- Detailed table printed to console
- Full report saved as CSV
- Immediate alerts for high-risk cases

---

## 🛠️ Technologies Used

- **Python 3.x** – Core language
- **`ast` module** – Code parsing and analysis
- **`networkx` + `matplotlib`** – Interactive network visualization
- **`seaborn` + `pandas`** – Beautiful statistical plots
- **`sklearn` (TF-IDF)** – Machine learning text analysis
- **`difflib`** – Sequence similarity detection
- **`collections.Counter`** – Feature frequency tracking

---

## ⚙️ Setup & Usage

### 1. Prepare Your Environment
```bash
git clone https://github.com/your-username/code-plagiarism-detector.git
cd code-plagiarism-detector
pip install -r requirements.txt
```

📝 **Note:** Create a folder named `homeworks/` and place all student `.py` files inside.

---

### 2. Run the Detector

```bash
python main.py
```

## That’s it! The tool will:

- ✅ Analyze all code pairs
- 📋 Print a full report
- 💾 Save results to `advanced_plagiarism_results.csv`
- 📊 Display interactive plots


## **🤝 Contribution Guidelines**

We welcome improvements! Feel free to:

- Add new similarity detection methods
- Improve AST normalization
- Support other languages (Java, C++, etc.)
- Enhance visualization aesthetics
- Add command-line arguments

Just open an issue or submit a pull request!

## 📄 License
- This project is licensed under the MIT License – see the [LICENSE](https://github.com/hrnrxb/Advanced-Code-Plagiarism-Detection-Tool/main/LICENSE) file for details.

## 💡 Pro Tips for Teachers

- 📅 Run this after every major assignment  
- 💬 Use results as conversation starters, not automatic penalties  
- 🔍 Look for clusters — they may indicate group work gone too far  
- 🗣️ Combine with oral exams for strongest evidence  
- 📁 Keep CSV reports for academic records  

---

## 🔎 Bottom Line  
This tool doesn’t just catch cheaters — it helps you **teach integrity** by providing clear, objective evidence of code similarity.

> Let the machine do the grunt work. You focus on teaching. 🤓

<br>

<div align="center">
  <strong>🌟 Stay ahead of plagiarism. Stay fair. Stay informed.</strong>
</div>

