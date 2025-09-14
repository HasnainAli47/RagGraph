import json
import os
from datetime import datetime
import requests
import streamlit as st

class SimpleLogger:
    def __init__(self, log_file="user_logs.json"):
        self.log_file = log_file
        self.ensure_log_file()
    
    def ensure_log_file(self):
        """Create log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def get_user_ip(self):
        """Get user's IP address"""
        try:
            # Try to get IP from Streamlit session state first
            if hasattr(st.session_state, 'user_ip'):
                return st.session_state.user_ip
            
            # Fallback: try to get from request headers
            import streamlit.web.server.server as server
            if hasattr(server, 'get_current_request'):
                request = server.get_current_request()
                if request:
                    # Try different header fields
                    ip = (request.headers.get('X-Forwarded-For') or 
                          request.headers.get('X-Real-IP') or 
                          request.headers.get('Remote-Addr') or 
                          '127.0.0.1')
                    st.session_state.user_ip = ip
                    return ip
        except:
            pass
        
        # Default fallback
        return '127.0.0.1'
    
    def get_location_from_ip(self, ip):
        """Get location from IP address using a free service"""
        try:
            if ip == '127.0.0.1' or ip.startswith('192.168.') or ip.startswith('10.'):
                return "Local Network"
            
            # Use ipapi.co (free tier: 1000 requests/day)
            response = requests.get(f"http://ipapi.co/{ip}/json/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                city = data.get('city', 'Unknown')
                country = data.get('country_name', 'Unknown')
                return f"{city}, {country}"
        except:
            pass
        
        return "Unknown Location"
    
    def get_user_agent(self):
        """Get user agent string"""
        try:
            import streamlit.web.server.server as server
            if hasattr(server, 'get_current_request'):
                request = server.get_current_request()
                if request:
                    return request.headers.get('User-Agent', 'Unknown')
        except:
            pass
        return 'Unknown'
    
    def log_user_session(self, action, details=None):
        """Log user session with IP, location, and action"""
        try:
            # Get user information
            ip = self.get_user_ip()
            location = self.get_location_from_ip(ip)
            user_agent = self.get_user_agent()
            timestamp = datetime.now().isoformat()
            
            # Create log entry
            log_entry = {
                "timestamp": timestamp,
                "ip_address": ip,
                "location": location,
                "user_agent": user_agent,
                "action": action,
                "details": details or {},
                "session_id": st.session_state.get('session_id', 'unknown')
            }
            
            # Read existing logs
            try:
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            except:
                logs = []
            
            # Add new log entry
            logs.append(log_entry)
            
            # Keep only last 1000 entries to prevent file from growing too large
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            # Silent fail - don't break the app if logging fails
            pass
    
    def log_search(self, search_type, content_preview, result_count=None):
        """Log user search activity"""
        details = {
            "search_type": search_type,  # "PDF" or "Text"
            "content_length": len(content_preview) if content_preview else 0,
            "content_preview": content_preview[:100] + "..." if content_preview and len(content_preview) > 100 else content_preview,
            "result_count": result_count
        }
        self.log_user_session("search", details)
    
    def log_graph_generation(self, triplets_count, nodes_count, edges_count):
        """Log successful graph generation"""
        details = {
            "triplets_extracted": triplets_count,
            "graph_nodes": nodes_count,
            "graph_edges": edges_count
        }
        self.log_user_session("graph_generated", details)
    
    def log_error(self, error_type, error_message):
        """Log errors"""
        details = {
            "error_type": error_type,
            "error_message": str(error_message)[:200]  # Limit error message length
        }
        self.log_user_session("error", details)
    
    def get_logs_summary(self):
        """Get a summary of logs for admin view"""
        try:
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            if not logs:
                return {"total_sessions": 0, "unique_ips": 0, "total_searches": 0}
            
            unique_ips = len(set(log['ip_address'] for log in logs))
            total_searches = len([log for log in logs if log['action'] == 'search'])
            total_graphs = len([log for log in logs if log['action'] == 'graph_generated'])
            
            return {
                "total_sessions": len(logs),
                "unique_ips": unique_ips,
                "total_searches": total_searches,
                "total_graphs_generated": total_graphs,
                "recent_activity": logs[-10:] if logs else []
            }
        except:
            return {"total_sessions": 0, "unique_ips": 0, "total_searches": 0}

# Initialize logger
logger = SimpleLogger()
