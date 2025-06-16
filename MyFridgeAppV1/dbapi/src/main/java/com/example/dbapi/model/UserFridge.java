package com.example.dbapi.model;

import jakarta.persistence.*;
import java.time.LocalDate;

@Entity
public class UserFridge 
{
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "user_id", nullable = false)
    private int userId;

    private int amount;

    private LocalDate eklenmeTarihi;

    private String skt;

    @ManyToOne
    @JoinColumn(name = "product_id")
    private Product product;

    public UserFridge() {}

    public UserFridge(int userId, Product product, int amount, String skt) 
    {
        this.userId = userId;
        this.product = product;
        this.amount = amount;
        this.eklenmeTarihi = LocalDate.now();
        this.skt = skt;
    }

    // Getter & Setter
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public int getUserId() { return userId; }
    public void setUserId(int userId) { this.userId = userId; }

    public int getAmount() { return amount; }
    public void setAmount(int amount) { this.amount = amount; }

    public Product getProduct() { return product; }
    public void setProduct(Product product) { this.product = product; }

    public LocalDate getEklenmeTarihi() { return eklenmeTarihi; }
    public void setEklenmeTarihi(LocalDate eklenmeTarihi) { this.eklenmeTarihi = eklenmeTarihi; }

    public String getSkt() { return skt; }
    public void setSkt(String skt) { this.skt = skt; }

    @Override
    public String toString() 
    {
        return "UserFridge{" +
                "id=" + id +
                ", userId=" + userId +
                ", amount=" + amount +
                ", eklenmeTarihi=" + eklenmeTarihi +
                ", sonKullanmaTarihi=" + skt +
                ", product=" + product +
                '}';
    }
}
