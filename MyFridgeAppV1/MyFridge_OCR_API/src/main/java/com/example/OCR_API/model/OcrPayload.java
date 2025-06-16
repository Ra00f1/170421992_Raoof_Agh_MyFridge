package com.example.OCR_API.model;
import java.util.List;

public class OcrPayload 
{
    private int userid;
    private List<List<String>> items;

    public int getUserid() 
    { 
        return userid; 
    }
    public void setUserid(int userid) 
    { 
        this.userid = userid; 
    }

    public List<List<String>> getItems() 
    { 
        return items; 
    }
    public void setItems(List<List<String>> items) 
    { 
        this.items = items; 
    }
}
