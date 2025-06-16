package com.example.mainapi.dto;

import java.util.List;

public class OcrProductDto 
{
    private Long userid;
    private List<List<String>> items;

    public Long getUserid() 
    {
        return userid;
    }

    public void setUserid(Long userid) 
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
