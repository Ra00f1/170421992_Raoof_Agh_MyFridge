package com.example.mainapi.model;

public class ProductTransferDto 
{
    private String categoryName;
    private String productName;
    private double price;
    private int amount;
    private Long userId;
    private String countType;
    private String skt;

    // Getters and Setters
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

    public double getPrice() 
    { 
        return price; 
    }
    public void setPrice(double price) 
    { 
        this.price = price; 
    }

    public int getAmount() 
    { 
        return amount; 
    }
    public void setAmount(int amount) 
    { 
        this.amount = amount; 
    }

    public Long getUserId() 
    { 
        return userId; 
    }
    public void setUserId(Long userId) 
    { 
        this.userId = userId; 
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

    @Override
    public String toString() {
        return "ProductTransferDto{" +
                "categoryName='" + categoryName + '\'' +
                ", productName='" + productName + '\'' +
                ", price=" + price +
                ", amount=" + amount +
                ", userId=" + userId +
                ", countType='" + countType + '\'' +
                ", skt='" + skt + '\'' +
                '}';
    }
}
