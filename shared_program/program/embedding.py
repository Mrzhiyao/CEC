# Re-export from parent module to maintain internal import compatibility
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from embedding import embedding
