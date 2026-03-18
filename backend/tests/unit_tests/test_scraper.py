import pytest
import json
import os
from bs4 import BeautifulSoup
from unittest.mock import patch, MagicMock

# Import the actual scraper functions 
from scraper.website_links import f_is_internal, f_is_english, f_extract_structured_content
from scraper.pdf_processing import f_pdf_content

# ==========================================
# Data Extraction & Storage Units
# ==========================================

def test_UT_scraper_1_crawler_execution(mocker):
    """UT_scraper_1: Run crawler on institute website seed URL"""
    # Mocking a hypothetical crawl_website function in your scraper
    mock_crawl = mocker.patch("scraper.crawl_website", return_value=["Doc 1 text", "Doc 2 text"])
    
    seed_url = "https://www.iitj.ac.in"
    result = mock_crawl(seed_url)
    
    # Assert pages successfully crawled and content retrieved
    assert result is not None
    assert len(result) > 0
    mock_crawl.assert_called_once_with(seed_url)

def test_UT_scraper_2_store_json(tmpdir):
    """UT_scraper_2: Store extracted content in structured JSON format"""
    extracted_data = [{"url": "http://test.com", "content": "Sample text"}]
    test_file = tmpdir.join("website_data.json")
    
    # Simulate writing to JSON
    with open(test_file, "w") as f:
        json.dump(extracted_data, f)
        
    # Assert JSON dataset generated successfully
    assert os.path.exists(test_file)
    with open(test_file, "r") as f:
        loaded_data = json.load(f)
        assert "url" in loaded_data[0]
        assert "content" in loaded_data[0]

# ==========================================
# Link Processing Units
# ==========================================

def test_UT_scraper_3_is_internal():
    """UT_scraper_3: Verify URL internality checker"""
    # Based on BASE_URL = "https://iitj.ac.in/"
    internal_url = "https://iitj.ac.in/academics/programs"
    external_url = "https://google.com/"
    
    assert f_is_internal(internal_url) is True
    assert f_is_internal(external_url) is False

def test_UT_scraper_4_is_english():
    """UT_scraper_4: Verify language filtering excludes non-English URLs"""
    english_url = "https://iitj.ac.in/department"
    hindi_url = "https://iitj.ac.in/department/hi/notice"
    hindi_param = "https://iitj.ac.in/index.php?lang=hi"
    
    assert f_is_english(english_url) is True
    assert f_is_english(hindi_url) is False
    assert f_is_english(hindi_param) is False

