package com.example.mainapi.dto;

public class EksiltmeDto 
{
    private Long userId;
    private String productName;
    private int miktar;

    // Constructor, Getter, Setter

    public EksiltmeDto() 
    {
        
    }

    public EksiltmeDto(Long userId, String productName, int miktar) 
    {
        this.userId = userId;
        this.productName = productName;
        this.miktar = miktar;
    }

    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }

    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }

    public int getMiktar() { return miktar; }
    public void setMiktar(int miktar) { this.miktar = miktar; }
}
