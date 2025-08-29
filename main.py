import ast
from collections import Counter
from itertools import combinations
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# Set better matplotlib parameters for larger, clearer plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# -----------------------------
# Convert code to AST features (improved version)
# -----------------------------
def canonicalize_code_ast(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()  # return empty counter if code is broken
    
    class NormalizeNames(ast.NodeTransformer):
        def __init__(self):
            self.var_count = 0
            self.func_count = 0
            self.name_mapping = {}
            # keep built-in function names unchanged
            self.builtin_names = {'print', 'len', 'range', 'int', 'str', 'float', 'list', 'dict', 'set', 'tuple'}
            
        def visit_Name(self, node):
            if isinstance(node.ctx, (ast.Load, ast.Store, ast.Param)):
                if node.id not in self.builtin_names and node.id not in self.name_mapping:
                    self.name_mapping[node.id] = f"var_{self.var_count}"
                    self.var_count += 1
                if node.id not in self.builtin_names:
                    node.id = self.name_mapping.get(node.id, node.id)
            return node
            
        def visit_arg(self, node):
            if node.arg not in self.name_mapping:
                self.name_mapping[node.arg] = f"arg_{self.var_count}"
                self.var_count += 1
            node.arg = self.name_mapping[node.arg]
            return node
            
        def visit_FunctionDef(self, node):
            if node.name not in self.name_mapping:
                self.name_mapping[node.name] = f"func_{self.func_count}"
                self.func_count += 1
            node.name = self.name_mapping[node.name]
            return self.generic_visit(node)
    
    tree = NormalizeNames().visit(tree)
    ast.fix_missing_locations(tree)
    
    # extract different AST features
    features = Counter()
    
    for node in ast.walk(tree):
        node_type = type(node).__name__
        features[f"node_{node_type}"] += 1
        
        # specific features for better detection
        if isinstance(node, ast.Name):
            features[f"name_{node.id}"] += 1
        elif isinstance(node, ast.Constant):
            features[f"const_{type(node.value).__name__}"] += 1
        elif isinstance(node, ast.BinOp):
            features[f"binop_{type(node.op).__name__}"] += 1
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                features[f"comp_{type(op).__name__}"] += 1
                
    return features

# -----------------------------
# Calculate similarity using different methods
# -----------------------------
def jaccard_similarity(counter1, counter2):
    """Jaccard similarity - more precise for copy detection"""
    all_keys = set(counter1.keys()).union(counter2.keys())
    intersect = sum(min(counter1[k], counter2[k]) for k in all_keys)
    union = sum(max(counter1[k], counter2[k]) for k in all_keys)
    return 100 * intersect / union if union else 0

def cosine_similarity(counter1, counter2):
    """Cosine similarity - more robust against small changes"""
    all_keys = set(counter1.keys()).union(counter2.keys())
    dot_product = sum(counter1[k] * counter2[k] for k in all_keys)
    norm1 = sum(counter1[k] ** 2 for k in all_keys) ** 0.5
    norm2 = sum(counter2[k] ** 2 for k in all_keys) ** 0.5
    return 100 * dot_product / (norm1 * norm2) if norm1 * norm2 else 0

def structural_similarity(counter1, counter2):
    """Structural similarity - only focuses on node types"""
    struct1 = Counter({k: v for k, v in counter1.items() if k.startswith('node_')})
    struct2 = Counter({k: v for k, v in counter2.items() if k.startswith('node_')})
    return jaccard_similarity(struct1, struct2)

def tf_idf_similarity(code1, code2):
    """TF-IDF based similarity using machine learning approach"""
    # convert codes to string for TF-IDF processing
    vectorizer = TfidfVectorizer(
        token_pattern=r'\b\w+\b',
        ngram_range=(1, 3),
        max_features=1000,
        stop_words=None
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform([code1, code2])
        similarity_matrix = sklearn_cosine_similarity(tfidf_matrix)
        return similarity_matrix[0, 1] * 100
    except:
        return 0

def sequence_similarity(code1, code2):
    """Sequence-based similarity using character matching"""
    # remove spaces and normalize
    clean_code1 = ''.join(code1.split())
    clean_code2 = ''.join(code2.split())
    
    matcher = SequenceMatcher(None, clean_code1, clean_code2)
    return matcher.ratio() * 100

# -----------------------------
# Load student code files
# -----------------------------
def load_student_codes(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
    codes = []
    student_names = []
    
    for file in files:
        try:
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():  # only non-empty files
                    codes.append(content)
                    student_names.append(file.replace(".py", ""))
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            
    return codes, student_names

# -----------------------------
# Main plagiarism analysis
# -----------------------------
def analyze_plagiarism(folder_path, threshold_jaccard=60, threshold_cosine=65, 
                      threshold_structural=70, threshold_tfidf=70, threshold_sequence=75):
    codes, student_names = load_student_codes(folder_path)
    
    if len(codes) < 2:
        print("At least two files are required!")
        return pd.DataFrame(), None
        
    print(f"Files loaded: {len(codes)}")
    
    # calculate feature counters for all codes
    counters = [canonicalize_code_ast(code) for code in codes]
    
    # calculate similarities using all methods
    results = []
    for i, j in combinations(range(len(codes)), 2):
        jaccard_sim = jaccard_similarity(counters[i], counters[j])
        cosine_sim = cosine_similarity(counters[i], counters[j])
        struct_sim = structural_similarity(counters[i], counters[j])
        tfidf_sim = tf_idf_similarity(codes[i], codes[j])
        seq_sim = sequence_similarity(codes[i], codes[j])
        
        # combined score with optimized weights
        combined_score = (jaccard_sim * 0.25 + cosine_sim * 0.25 + struct_sim * 0.15 + 
                         tfidf_sim * 0.25 + seq_sim * 0.10)
        
        # determine suspicion level with improved criteria
        if (jaccard_sim >= threshold_jaccard or 
            cosine_sim >= threshold_cosine or 
            struct_sim >= threshold_structural or
            tfidf_sim >= threshold_tfidf or
            seq_sim >= threshold_sequence):
            
            if combined_score >= 80:
                suspicion_level = "Very High"
                color_code = "🔴"
            elif combined_score >= 70:
                suspicion_level = "High"
                color_code = "🟠"
            elif combined_score >= 60:
                suspicion_level = "Moderate"
                color_code = "🟡"
            else:
                suspicion_level = "Low"
                color_code = "🟢"
                
            results.append({
                "Student 1": student_names[i],
                "Student 2": student_names[j], 
                "Jaccard %": round(jaccard_sim, 1),
                "Cosine %": round(cosine_sim, 1),
                "Structural %": round(struct_sim, 1),
                "TF-IDF %": round(tfidf_sim, 1),
                "Sequence %": round(seq_sim, 1),
                "Combined Score": round(combined_score, 1),
                "Suspicion Level": suspicion_level,
                "Risk": color_code
            })
    
    # sort by combined score (highest first)
    results.sort(key=lambda x: x["Combined Score"], reverse=True)
    df = pd.DataFrame(results)
    
    return df, (codes, student_names, counters)

# -----------------------------
# Display complete results table
# -----------------------------
def display_results(df):
    if df.empty:
        print("No suspicious cases found!")
        return
        
    print("Complete Plagiarism Analysis Results:")
    print("=" * 120)
    
    # show full table without row limits
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df.to_string(index=False))
    
    print(f"\nStatistical Summary:")
    print(f"Total suspicious pairs: {len(df)}")
    print(f"Average combined score: {df['Combined Score'].mean():.1f}")
    print(f"Highest score: {df['Combined Score'].max():.1f}")
    print(f"Lowest score: {df['Combined Score'].min():.1f}")
    
    # show statistics for each method
    print(f"\nMethod Statistics:")
    for method in ['Jaccard %', 'Cosine %', 'Structural %', 'TF-IDF %', 'Sequence %']:
        print(f"{method}: Max={df[method].max():.1f}, Avg={df[method].mean():.1f}")

# -----------------------------
# Enhanced network plot with better spacing
# -----------------------------
def create_network_plot(df, student_names):
    if df.empty:
        print("No data available for network plot!")
        return
        
    # create larger figure with better spacing
    fig = plt.figure(figsize=(22, 16))
    
    G = nx.Graph()
    G.add_nodes_from(student_names)
    
    # add edges with weights
    for _, row in df.iterrows():
        G.add_edge(row["Student 1"], row["Student 2"], 
                  weight=row["Combined Score"],
                  jaccard=row["Jaccard %"],
                  cosine=row["Cosine %"],
                  structural=row["Structural %"],
                  tfidf=row["TF-IDF %"],
                  sequence=row["Sequence %"])
    
    # calculate layout with more spacing
    pos = nx.spring_layout(G, k=3, iterations=150, seed=42)
    
    # color nodes based on connection count
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        connections = len(list(G.neighbors(node)))
        if connections >= 4:
            node_colors.append('darkred')
            node_sizes.append(3500)
        elif connections >= 3:
            node_colors.append('red')
            node_sizes.append(3000)
        elif connections >= 2:
            node_colors.append('orange') 
            node_sizes.append(2500)
        elif connections >= 1:
            node_colors.append('yellow')
            node_sizes.append(2000)
        else:
            node_colors.append('lightblue')
            node_sizes.append(1500)
    
    # edge colors and widths based on similarity scores
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        weight = G[u][v]['weight']
        if weight >= 80:
            edge_colors.append('darkred')
            edge_widths.append(6)
        elif weight >= 70:
            edge_colors.append('red')
            edge_widths.append(5)
        elif weight >= 60:
            edge_colors.append('orange')
            edge_widths.append(4)
        elif weight >= 50:
            edge_colors.append('gold')
            edge_widths.append(3)
        else:
            edge_colors.append('gray')
            edge_widths.append(2)
    
    # draw the network
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
            width=edge_widths,
            font_size=11,
            font_weight='bold',
            alpha=0.9)
    
    # add edge labels with similarity scores
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        edge_labels[(u, v)] = f"{d['weight']:.0f}%"
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, font_weight='bold')
    
    plt.title("Advanced Code Plagiarism Detection Network\n" + 
             "Dark Red: Critical Risk | Red: High Risk | Orange: Medium Risk | Yellow: Low Risk", 
             fontsize=16, pad=20)
    
    # improved legend with better positioning
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
                   markersize=18, label='>=4 Connections (Critical)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=16, label='3 Connections (High Risk)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                   markersize=14, label='2 Connections (Medium Risk)'), 
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                   markersize=12, label='1 Connection (Low Risk)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='No Suspicious Connection')
    ]
    
    # position legend outside plot area to avoid overlap
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=11)
    
    # add statistics box in a better position
    stats_text = f"Total Students: {len(student_names)}\n"
    stats_text += f"Suspicious Pairs: {len(df)}\n"
    stats_text += f"Avg Score: {df['Combined Score'].mean():.1f}%\n"
    stats_text += f"Max Score: {df['Combined Score'].max():.1f}%"
    
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom', fontsize=10)
    
    # adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.show()

# -----------------------------
# Enhanced statistical plots with better spacing
# -----------------------------
def create_statistics_plots(df):
    if df.empty:
        return
        
    # create larger figure with improved spacing
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.4, 
                         left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # 1. Combined score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df['Combined Score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df['Combined Score'].mean(), color='red', linestyle='--', 
               label=f'Average: {df["Combined Score"].mean():.1f}')
    ax1.set_title('Combined Score Distribution', fontsize=13, pad=15)
    ax1.set_xlabel('Combined Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. All methods comparison
    ax2 = fig.add_subplot(gs[0, 1])
    methods = ['Jaccard %', 'Cosine %', 'Structural %', 'TF-IDF %', 'Sequence %']
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, method in enumerate(methods):
        ax2.scatter(df['Combined Score'], df[method], alpha=0.7, 
                   label=method.replace(' %', ''), color=colors[i], s=50)
    ax2.set_title('Similarity Methods vs Combined Score', fontsize=13, pad=15)
    ax2.set_xlabel('Combined Score')
    ax2.set_ylabel('Method Score')
    # position legend to avoid overlap
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation heatmap with better sizing
    ax3 = fig.add_subplot(gs[0, 2])
    corr_data = df[['Jaccard %', 'Cosine %', 'Structural %', 'TF-IDF %', 'Sequence %', 'Combined Score']]
    correlation_matrix = corr_data.corr()
    im = ax3.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(correlation_matrix.columns)))
    ax3.set_yticks(range(len(correlation_matrix.columns)))
    # shorter labels to prevent overlap
    short_labels = ['Jaccard', 'Cosine', 'Struct', 'TF-IDF', 'Sequence', 'Combined']
    ax3.set_xticklabels(short_labels, rotation=45, ha='right')
    ax3.set_yticklabels(short_labels)
    ax3.set_title('Methods Correlation Matrix', fontsize=13, pad=15)
    
    # add correlation values with better formatting
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            text = ax3.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=8)
    
    # 4. Suspicion level pie chart
    ax4 = fig.add_subplot(gs[1, 0])
    suspicion_counts = df['Suspicion Level'].value_counts()
    colors_pie = ['darkred', 'red', 'orange', 'yellow'][:len(suspicion_counts)]
    wedges, texts, autotexts = ax4.pie(suspicion_counts.values, labels=suspicion_counts.index, 
                                      autopct='%1.1f%%', colors=colors_pie, startangle=90)
    # make text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax4.set_title('Suspicion Level Distribution', fontsize=13, pad=15)
    
    # 5. Box plot for all methods with better spacing
    ax5 = fig.add_subplot(gs[1, 1])
    box_data = [df[method].values for method in methods]
    box_plot = ax5.boxplot(box_data, labels=[m.replace(' %', '') for m in methods], patch_artist=True)
    colors_box = ['lightcoral', 'lightblue', 'lightgreen', 'plum', 'lightsalmon']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
    ax5.set_title('Methods Score Distribution', fontsize=13, pad=15)
    ax5.set_ylabel('Score')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. Top suspicious pairs with better label handling
    ax6 = fig.add_subplot(gs[1, 2])
    top_10 = df.head(10)
    y_pos = range(len(top_10))
    # truncate long names to prevent overlap
    pair_labels = []
    for _, row in top_10.iterrows():
        name1 = row['Student 1'][:8] + '...' if len(row['Student 1']) > 10 else row['Student 1']
        name2 = row['Student 2'][:8] + '...' if len(row['Student 2']) > 10 else row['Student 2']
        pair_labels.append(f"{name1} - {name2}")
    
    bars = ax6.barh(y_pos, top_10['Combined Score'], 
                   color=['darkred' if x >= 80 else 'red' if x >= 70 else 'orange' 
                          for x in top_10['Combined Score']])
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(pair_labels, fontsize=9)
    ax6.set_title('Top 10 Most Suspicious Pairs', fontsize=13, pad=15)
    ax6.set_xlabel('Combined Score')
    ax6.grid(True, alpha=0.3)
    
    # add score values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax6.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontweight='bold', fontsize=8)
    
    plt.suptitle('Comprehensive Plagiarism Analysis Dashboard', fontsize=18, y=0.96)
    plt.show()

# -----------------------------
# Main execution function
# -----------------------------
def main():
    folder_path = "homeworks"  # folder containing student files
    
    print("🔍 Advanced Code Plagiarism Detection Starting...")
    df, data = analyze_plagiarism(folder_path)
    
    # display results
    display_results(df)
    
    if data and not df.empty:
        codes, student_names, counters = data
        
        print(f"\n🎯 Detection Methods Used:")
        print("1. Jaccard Similarity - Direct copy detection")
        print("2. Cosine Similarity - Structural similarity")
        print("3. Structural AST - Code structure analysis")
        print("4. TF-IDF ML - Machine learning text similarity")
        print("5. Sequence Matching - Character-level similarity")
        
        # create visualizations
        create_network_plot(df, student_names)
        create_statistics_plots(df)
        
        # save results
        df.to_csv("advanced_plagiarism_results.csv", index=False, encoding="utf-8-sig")
        print(f"\n💾 Results saved to: advanced_plagiarism_results.csv")
        
        # final summary
        very_high = len(df[df['Suspicion Level'] == 'Very High'])
        high = len(df[df['Suspicion Level'] == 'High'])
        moderate = len(df[df['Suspicion Level'] == 'Moderate'])
        low = len(df[df['Suspicion Level'] == 'Low'])
        
        print(f"\n📊 Final Summary:")
        print(f"Very High Risk: {very_high} pairs")
        print(f"High Risk: {high} pairs")
        print(f"Moderate Risk: {moderate} pairs")
        print(f"Low Risk: {low} pairs")
        
        if very_high > 0:
            print(f"\n⚠️  IMMEDIATE ACTION REQUIRED for {very_high} high-risk cases!")

# run the program
if __name__ == "__main__":
    main()
