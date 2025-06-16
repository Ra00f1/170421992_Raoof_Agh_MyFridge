package com.example.dbapi.dto;

public class EksiltmeDto 
{
    private Long userId;
    private String productName;
    private int miktar;

    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }

    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }

    public int getMiktar() { return miktar; }
    public void setMiktar(int miktar) { this.miktar = miktar; }
}