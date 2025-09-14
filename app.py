import streamlit as st
import os
import networkx as nx
import random
from pyvis.network import Network
import tempfile
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
from datetime import datetime
from logger import logger

# --- CONFIGURATION & PAGE SETUP ---
load_dotenv()  # For local development
MAX_FILE_SIZE_MB = 2
MAX_TEXT_CHARS = 6000

# Get API key from environment or Streamlit secrets
def get_api_key():
    # Try Streamlit secrets first (for deployment)
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        # Fall back to environment variable (for local development)
        return os.environ.get("GROQ_API_KEY")

st.set_page_config(
    page_title="GraphifyAI",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJECT ROBUST CSS FOR PROFESSIONAL UI (FINAL VERSION) ---
st.markdown("""
<style>
    /* --- Main App Styles --- */
    .stApp {
        background-color: #F0F2F6;
    }
    .stApp, .stApp div, .stApp p, .stApp li {
        color: #31333F; /* Default dark text for main area */
    }
    h1 { color: #1E293B; }
    h2 {
        color: #334155;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 8px;
    }
    .st-emotion-cache-1v0evarda { /* Streamlit's bordered container */
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }

    /* --- Sidebar Styling (FINAL FIX) --- */
    [data-testid="stSidebar"] {
        background-color: #0F172A; /* Dark Slate background */
    }

    /* Set default text color for all content within the sidebar */
    [data-testid="stSidebar"] > div:first-child {
        color: #FFFFFF; /* PURE WHITE for body text as requested */
    }

    /* Style for headers within the sidebar */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF; /* Bright white for headers */
        border-bottom: none;
    }

    /* This rule ensures text inside st.success/st.error remains dark */
    [data-testid="stSidebar"] .stAlert p {
        color: #0F172A; /* Dark text for alerts inside the sidebar */
    }
</style>
""", unsafe_allow_html=True)


# --- CORE RAG-GRAPH FUNCTIONS (Cached for performance) ---

def process_input(content, input_type):
    """Unified function to chunk either PDF content or raw text."""
    if input_type == "PDF":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load()
        os.remove(tmp_file_path)
        full_text = "".join(page.page_content for page in pages if page.page_content)
    else: # input_type == "Text"
        full_text = content
        
    if not full_text: return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(full_text)

def get_extraction_chain(_api_key):
    """Creates the LangChain extraction chain."""
    graph_schema = {
        "entity_types": ["Model", "Technique", "Component", "Metric", "Dataset", "Paper", "Author", "Organization"],
        "relation_types": ["USES", "IMPROVES", "EVALUATED_ON", "COMPARED_TO", "PART_OF", "INTRODUCED_IN", "AUTHOR_OF", "AFFILIATED_WITH"]
    }
    prompt_template = """
     You are an expert at extracting knowledge triplets from text. Your task is to identify entities and their relationships.

     IMPORTANT: Always return a valid JSON response, even if you find only one triplet.

     Entity Types: {entity_types}
     Relationship Types: {relation_types}

     Instructions:
     1. Extract meaningful relationships between entities
     2. Use the exact entity and relationship types from the lists above
     3. If you find any entities or relationships, return them in the JSON format
     4. If no clear relationships exist, try to extract at least entity mentions

     Example:
     Text: "BERT is a language model that uses attention mechanisms and was developed by Google."
     
     Response:
     {{
       "triplets": [
         {{
           "subject": {{"name": "BERT", "type": "Model"}},
           "relation": "USES",
           "object": {{"name": "attention mechanisms", "type": "Technique"}}
         }},
         {{
           "subject": {{"name": "BERT", "type": "Model"}},
           "relation": "AUTHOR_OF",
           "object": {{"name": "Google", "type": "Organization"}}
         }}
       ]
     }}

     Text to analyze:
     {text_chunk}

     Remember: Always return valid JSON with at least one triplet if possible.
     """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=_api_key)
    json_parser = JsonOutputParser()
    return prompt | llm | json_parser

def extract_triplets(chunks, api_key):
    """Extracts triplets with progress tracking."""
    all_triplets = []
    extraction_chain = get_extraction_chain(api_key)
    
    # Create progress bar
    progress_bar = st.progress(0, text="Extracting knowledge triplets...")
    
    for i, chunk in enumerate(chunks):
        try:
            # Update progress
            progress_bar.progress((i + 1) / len(chunks), text=f"Processing chunk {i+1}/{len(chunks)}")
            
            extracted_data = extraction_chain.invoke({
                "text_chunk": chunk,
                "entity_types": ["Model", "Technique", "Component", "Metric", "Dataset", "Paper", "Author", "Organization"],
                "relation_types": ["USES", "IMPROVES", "EVALUATED_ON", "COMPARED_TO", "PART_OF", "INTRODUCED_IN", "AUTHOR_OF", "AFFILIATED_WITH"]
            })
            
            if extracted_data and 'triplets' in extracted_data and extracted_data['triplets']:
                all_triplets.extend(extracted_data['triplets'])
                st.write(f"✅ Chunk {i+1}: Found {len(extracted_data['triplets'])} triplets")
            else:
                st.write(f"⚠️ Chunk {i+1}: No triplets extracted")
                
        except Exception as e:
            st.write(f"❌ Chunk {i+1}: Error - {str(e)}")
            logger.log_error("chunk_processing_error", f"Chunk {i+1}: {str(e)}")
    
    progress_bar.empty()
    return all_triplets

def build_graph(triplets):
    """Builds a networkx graph from the extracted triplets."""
    G = nx.DiGraph()
    for triplet in triplets:
        subject_info = triplet.get('subject', {})
        object_info = triplet.get('object', {})
        if subject_info.get('name') and object_info.get('name'):
            G.add_node(subject_info['name'], type=subject_info.get('type'))
            G.add_node(object_info['name'], type=object_info.get('type'))
            G.add_edge(subject_info['name'], object_info['name'], label=triplet.get('relation'))
    return G

def generate_graph_html(graph):
    """Generates an improved interactive HTML graph visualization."""
    entity_types = list(set(nx.get_node_attributes(graph, 'type').values()))
    colors = ["#FFC0CB", "#ADD8E6", "#90EE90", "#FFD700", "#DDA0DD", "#FA8072", "#B0C4DE", "#87CEEB"]
    color_map = dict(zip(entity_types, colors[:len(entity_types)]))
    net = Network(height="750px", width="100%", cdn_resources='in_line', directed=True, notebook=True)
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.1,
          "springLength": 150,
          "springConstant": 0.05
        },
        "minVelocity": 0.75
      }
    }
    """)
    degrees = dict(graph.degree())
    min_size, max_size = 15, 50
    for node, attrs in graph.nodes(data=True):
        node_type = attrs.get('type', 'Unknown')
        node_size = min(max(degrees.get(node, 1) * 5, min_size), max_size)
        net.add_node(node, label=node, title=f"Type: {node_type}<br>Connections: {degrees.get(node, 0)}", color=color_map.get(node_type, 'grey'), size=node_size)
    for u, v, attrs in graph.edges(data=True):
        net.add_edge(u, v, label=attrs.get('label', ''))
    html_file_name = "knowledge_graph.html"
    net.show(html_file_name)
    return html_file_name

# --- INITIALIZE SESSION STATE ---
if 'graph_generated' not in st.session_state:
    st.session_state.graph_generated = False
if 'html_data' not in st.session_state:
    st.session_state.html_data = None
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]
    # Log new session
    logger.log_user_session("session_start")

# --- UI LAYOUT ---
st.title("🕸️ GraphifyAI: Automated Knowledge Graph Builder")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = get_api_key()
    if groq_api_key:
        st.success("API Key configured.", icon="✅")
    else:
        st.error("API Key not found.", icon="❌")
        st.markdown("**For deployment:** Set `GROQ_API_KEY` in Streamlit Secrets")
        st.markdown("**For local:** Create a `.env` file with `GROQ_API_KEY=your_key`")
    st.markdown("---")
    st.info("This app transforms text into a dynamic knowledge graph, perfect for analyzing complex documents.")
    
    # Admin panel for viewing logs (restricted access)
    admin_password = os.environ.get("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD", "")
    
    if admin_password:
        # Show admin login
        if st.checkbox("🔍 Admin Access"):
            entered_password = st.text_input("Admin Password", type="password", key="admin_pw")
            if entered_password == admin_password:
                st.success("✅ Admin access granted")
                st.markdown("### 📊 Usage Statistics")
                try:
                    summary = logger.get_logs_summary()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Sessions", summary.get("total_sessions", 0))
                        st.metric("Unique IPs", summary.get("unique_ips", 0))
                    with col2:
                        st.metric("Total Searches", summary.get("total_searches", 0))
                        st.metric("Graphs Generated", summary.get("total_graphs_generated", 0))
                    
                    # Show recent activity
                    if summary.get("recent_activity"):
                        st.markdown("### 📝 Recent Activity")
                        for activity in summary["recent_activity"][-5:]:
                            st.text(f"{activity['timestamp'][:19]} | {activity['ip_address']} | {activity['location']} | {activity['action']}")
                    
                    # Download complete logs
                    st.markdown("### 📥 Download Complete Logs")
                    try:
                        with open("user_logs.json", 'r') as f:
                            complete_logs = f.read()
                        st.download_button(
                            label="📄 Download Full Logs (JSON)",
                            data=complete_logs,
                            file_name=f"user_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            help="Download complete user logs with all details"
                        )
                    except Exception as e:
                        st.error(f"Could not read log file: {str(e)}")
                    
                    # Show detailed logs in expandable sections
                    st.markdown("### 📋 Detailed Logs")
                    try:
                        with open("user_logs.json", 'r') as f:
                            all_logs = json.load(f)
                        
                        # Group logs by action type
                        search_logs = [log for log in all_logs if log['action'] == 'search']
                        graph_logs = [log for log in all_logs if log['action'] == 'graph_generated']
                        error_logs = [log for log in all_logs if log['action'] == 'error']
                        
                        # Show search logs
                        if search_logs:
                            with st.expander(f"🔍 Search Logs ({len(search_logs)} entries)"):
                                for log in search_logs[-10:]:  # Show last 10
                                    st.json({
                                        "timestamp": log['timestamp'],
                                        "ip": log['ip_address'],
                                        "location": log['location'],
                                        "search_type": log['details'].get('search_type'),
                                        "content_preview": log['details'].get('content_preview', '')[:100]
                                    })
                        
                        # Show graph generation logs
                        if graph_logs:
                            with st.expander(f"🕸️ Graph Generation Logs ({len(graph_logs)} entries)"):
                                for log in graph_logs[-10:]:  # Show last 10
                                    st.json({
                                        "timestamp": log['timestamp'],
                                        "ip": log['ip_address'],
                                        "location": log['location'],
                                        "triplets": log['details'].get('triplets_extracted'),
                                        "nodes": log['details'].get('graph_nodes'),
                                        "edges": log['details'].get('graph_edges')
                                    })
                        
                        # Show error logs
                        if error_logs:
                            with st.expander(f"❌ Error Logs ({len(error_logs)} entries)"):
                                for log in error_logs[-10:]:  # Show last 10
                                    st.json({
                                        "timestamp": log['timestamp'],
                                        "ip": log['ip_address'],
                                        "location": log['location'],
                                        "error_type": log['details'].get('error_type'),
                                        "error_message": log['details'].get('error_message')
                                    })
                                    
                    except Exception as e:
                        st.error(f"Could not load detailed logs: {str(e)}")
                    
                    # Show log file location
                    st.markdown("### 📁 Log File Location")
                    st.code("user_logs.json", language="text")
                    st.info("💡 Logs are stored in the `user_logs.json` file on the server")
                    
                except Exception as e:
                    st.error(f"Could not load logs: {str(e)}")
            elif entered_password:
                st.error("❌ Incorrect password")
    else:
        # Fallback: Show basic stats without sensitive data
        if st.checkbox("📊 Show Basic Stats"):
            try:
                summary = logger.get_logs_summary()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Sessions", summary.get("total_sessions", 0))
                    st.metric("Total Searches", summary.get("total_searches", 0))
                with col2:
                    st.metric("Graphs Generated", summary.get("total_graphs_generated", 0))
                    st.metric("Success Rate", f"{(summary.get('total_graphs_generated', 0) / max(summary.get('total_searches', 1), 1) * 100):.1f}%")
            except Exception as e:
                st.error(f"Could not load stats: {str(e)}")

# --- MAIN CONTENT ---
if not groq_api_key:
    st.warning("Please configure your GROQ_API_KEY environment variable to proceed.")
    st.stop()

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    with st.container(border=True):
        st.header("1. Provide Input")
        input_tab1, input_tab2 = st.tabs(["📄 PDF Upload", "✍️ Text Input"])

        with input_tab1:
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help=f"Max size: {MAX_FILE_SIZE_MB}MB")
        with input_tab2:
            raw_text = st.text_area("Or paste your text here", height=250, max_chars=MAX_TEXT_CHARS, placeholder="Enter text...")

        process_button = st.button("Generate Knowledge Graph", type="primary", use_container_width=True)

# --- PROCESSING LOGIC ---
if process_button:
    st.session_state.graph_generated = False
    st.session_state.html_data = None
    input_content, input_type = (uploaded_file.getvalue(), "PDF") if uploaded_file else (raw_text, "Text") if raw_text else (None, None)
    
    is_valid = True
    if uploaded_file and uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"PDF size exceeds {MAX_FILE_SIZE_MB}MB limit.", icon="🚨")
        is_valid = False
    if not input_content:
        st.warning("Please provide an input.", icon="⚠️")
        is_valid = False

    if is_valid:
        # Log the search attempt
        content_preview = input_content[:200] if input_content else ""
        logger.log_search(input_type, content_preview)
        
        # Step 1: Process input
        with st.spinner("📄 Processing input..."):
            chunks = process_input(input_content, input_type)
        
        if chunks:
            st.success(f"✅ Successfully processed {len(chunks)} text chunks")
            
            # Step 2: Extract triplets
            st.info("🧠 AI is analyzing text and extracting knowledge triplets...")
            all_triplets = extract_triplets(chunks, groq_api_key)
            
            if all_triplets:
                st.success(f"✅ Successfully extracted {len(all_triplets)} knowledge triplets")
                
                # Step 3: Build graph
                with st.spinner("🕸️ Building knowledge graph..."):
        graph = build_graph(all_triplets)
        
                # Step 4: Generate visualization
                with st.spinner("🎨 Generating interactive visualization..."):
        html_file = generate_graph_html(graph)
        with open(html_file, 'r', encoding='utf-8') as f:
                        st.session_state.html_data = f.read()
                    st.session_state.graph_generated = True
                
                # Log successful graph generation
                logger.log_graph_generation(
                    len(all_triplets), 
                    graph.number_of_nodes(), 
                    graph.number_of_edges()
                )
                
                st.success("🎉 Knowledge graph generated successfully!")
            else:
                # Log failed extraction
                logger.log_error("extraction_failed", "No triplets extracted from input")
                st.error("❌ The AI could not extract any valid knowledge triplets from the input.", icon="🚨")
                st.info("💡 Try with different text or check if the content contains clear entities and relationships.")
        else:
            # Log failed processing
            logger.log_error("processing_failed", "Could not extract text from input")
            st.error("❌ Could not extract any text from the input. The PDF might be image-based or corrupted.", icon="🚨")

# --- DISPLAY OUTPUT ---
with col2:
    with st.container(border=True):
        st.header("2. Explore Your Knowledge Graph")
        if st.session_state.graph_generated:
            st.components.v1.html(st.session_state.html_data, height=750, scrolling=True)
            st.download_button("Download Graph HTML", st.session_state.html_data, "knowledge_graph.html", "text/html", use_container_width=True)
else:
            st.info("Your interactive graph will be displayed here once it's generated.")