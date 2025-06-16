package com.example.dbapi.repository;

import com.example.dbapi.model.Category;
import com.example.dbapi.model.Product;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

import java.util.Optional;

public interface ProductRepository extends JpaRepository<Product, Long> 
{
    Optional<Product> findByNameAndCategory(String name, Category category);

    Optional<Product> findByNameIgnoreCase(String name);

    Optional<Product> findByNameIgnoreCaseAndCountType(String name, String countType);

    List<Product> findAllByNameIgnoreCase(String name);

}