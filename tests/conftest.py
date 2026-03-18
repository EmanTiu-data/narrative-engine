# Test configuration
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from freezegun import freeze_time
import responses

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure pytest
pytest_plugins = ['pytest_asyncio']
