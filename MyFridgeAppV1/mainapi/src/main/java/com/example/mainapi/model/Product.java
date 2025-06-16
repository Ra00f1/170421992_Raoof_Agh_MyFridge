package com.example.mainapi.model;

public class Product 
{
    private String name;
    private String category;
    private String brand;
    private int amount;
    private double price;

    public Product(String name, String category, String brand, int amount, double price) 
    {
        this.name = name;
        this.category = category;
        this.brand = brand;
        this.amount = amount;
        this.price = price;
    }

    public String getName() 
    { 
        return name; 
    }
    public void setName(String name) 
    { 
        this.name = name; 
    }
    public String getCategory() 
    { 
        return category; 
    }
    public void setCategory(String category) 
    { 
        this.category = category; 
    }
    public String getBrand() 
    { 
        return brand; 
    }
    public void setBrand(String brand) 
    { 
        this.brand = brand; 
    }
    public int getAmount() 
    { 
        return amount; 
    }
    public void setAmount(int amount) 
    { 
        this.amount = amount; 
    }

    public double getPrice() { return price; }
    public void setPrice(double price) { this.price = price; }
}
