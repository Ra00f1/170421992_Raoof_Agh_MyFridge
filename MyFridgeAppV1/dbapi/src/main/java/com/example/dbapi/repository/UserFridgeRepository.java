package com.example.dbapi.repository;

import com.example.dbapi.model.UserFridge;
import com.example.dbapi.model.Product;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UserFridgeRepository extends JpaRepository<UserFridge, Long> 
{

    Optional<UserFridge> findByUserIdAndProduct(int userId, Product product);

    Optional<UserFridge> findByUserIdAndProductId(Integer userId, Long productId);

    // Kullanıcının tüm ürünlerini getir
    List<UserFridge> findAllByUserId(int userId);

    @Query("SELECT uf FROM UserFridge uf WHERE uf.userId = :userId AND uf.skt <= :today")
    List<UserFridge> findExpiringSoonByUserId(@Param("userId") int userId, @Param("today") String today);

    @Query("SELECT uf FROM UserFridge uf WHERE uf.userId = :userId AND uf.skt <= :targetDate")
    List<UserFridge> findExpiringIn3DaysByUserId(@Param("userId") int userId, @Param("targetDate") String targetDate);

    List<UserFridge> findAllByUserIdAndProduct_NameIgnoreCase(int userId, String productName);

    List<UserFridge> findAllByUserIdAndProduct(int userId, Product product);

}
