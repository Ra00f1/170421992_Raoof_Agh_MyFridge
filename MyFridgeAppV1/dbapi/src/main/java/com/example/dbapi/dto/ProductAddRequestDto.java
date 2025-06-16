package com.example.dbapi.dto;

public class ProductAddRequestDto 
{
    private int userId;
    private String categoryName;
    private String productName;
    private int amount;
    private double price;
    private String countType; 
    private String skt;

    // Getter ve Setter'lar

    public int getUserId() 
    {
        return userId;
    }

    public void setUserId(int userId) 
    {
        this.userId = userId;
    }

    public String getCategoryName() 
    {
        return categoryName;
    }

    public void setCategoryName(String categoryName) 
    {
        this.categoryName = categoryName;
    }

    public String getProductName() 
    {
        return productName;
    }

    public void setProductName(String productName) 
    {
        this.productName = productName;
    }

    public int getAmount() 
    {
        return amount;
    }

    public void setAmount(int amount) 
    {
        this.amount = amount;
    }

    public double getPrice() 
    {
        return price;
    }

    public void setPrice(double price) 
    {
        this.price = price;
    }
    public String getCountType() 
    {
        return countType;
    }
    public void setCountType(String countType) 
    {
        this.countType = countType;
    }
    public String getSkt() 
    {
        return skt;
    }
    public void setSkt(String skt) 
    {
        this.skt = skt;
    }
}