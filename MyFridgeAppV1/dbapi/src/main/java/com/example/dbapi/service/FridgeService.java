package com.example.dbapi.service;

import com.example.dbapi.model.Category;
import com.example.dbapi.model.Product;
import com.example.dbapi.model.UserFridge;
import com.example.dbapi.repository.CategoryRepository;
import com.example.dbapi.repository.ProductRepository;
import com.example.dbapi.repository.UserFridgeRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Optional;

@Service
public class FridgeService 
{

    private final CategoryRepository categoryRepository;
    private final ProductRepository productRepository;
    private final UserFridgeRepository userFridgeRepository;

    public FridgeService(CategoryRepository categoryRepository,
                         ProductRepository productRepository,
                         UserFridgeRepository userFridgeRepository) {
        this.categoryRepository = categoryRepository;
        this.productRepository = productRepository;
        this.userFridgeRepository = userFridgeRepository;
    }

    public List<UserFridge> getAllProductsForUser(int userId) {
        return userFridgeRepository.findAllByUserId(userId);
    }

    public void addProductToUserFridge(int userId, String categoryName, String productName,
                                       int amount, double price, String countType, String skt) {
        // 1. Kategori varsa al, yoksa oluştur
        Category category = categoryRepository.findByNameIgnoreCase(categoryName)
                .orElseGet(() -> categoryRepository.save(new Category(categoryName)));

        // 2. Ürün varsa al, yoksa oluştur
        List<Product> existingProducts = productRepository.findAllByNameIgnoreCase(productName);
        Product product = null;

        for (Product p : existingProducts) {
            if (p.getPrice() == price && p.getCountType().equalsIgnoreCase(countType)) {
                product = p;
                break;
            }
        }

        if (product == null) {
            product = new Product(productName, category.getName(), price, countType);
            product = productRepository.save(product);
        }

        // 3. Kullanıcının dolabında varsa miktarı arttır, yoksa yeni ekle
        Optional<UserFridge> existingUfOpt = userFridgeRepository.findByUserIdAndProduct(userId, product);
        if (existingUfOpt.isPresent()) {
            UserFridge existingUf = existingUfOpt.get();
            existingUf.setAmount(existingUf.getAmount() + amount);
            userFridgeRepository.save(existingUf);
        } else {
            UserFridge newUf = new UserFridge(userId, product, amount, skt);
            userFridgeRepository.save(newUf);
        }
    }

    public List<UserFridge> getExpiringProductsForUser(int userId) {
        String today = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        return userFridgeRepository.findExpiringSoonByUserId(userId, today);
    }

    public List<UserFridge> getExpiringSoonIn3Days(int userId) {
        String targetDate = LocalDate.now().plusDays(3).format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
        return userFridgeRepository.findExpiringIn3DaysByUserId(userId, targetDate);
    }

    public boolean deleteProduct(Long userId, String productName) {
        List<Product> productList = productRepository.findAllByNameIgnoreCase(productName);
        for (Product product : productList) {
            Optional<UserFridge> userFridgeOpt = userFridgeRepository.findByUserIdAndProduct(userId.intValue(), product);
            if (userFridgeOpt.isPresent()) {
                userFridgeRepository.delete(userFridgeOpt.get());
                return true;
            }
        }
        return false;
    }

    public void eksiltProduct(Long userId, String productName, int miktar) {
        List<Product> productList = productRepository.findAllByNameIgnoreCase(productName);
        if (productList.isEmpty()) {
            throw new RuntimeException("Ürün bulunamadı: " + productName);
        }

        for (Product product : productList) {
            List<UserFridge> ufList = userFridgeRepository.findAllByUserIdAndProduct(userId.intValue(), product);
            for (UserFridge uf : ufList) {
                int yeniMiktar = uf.getAmount() - miktar;
                if (yeniMiktar < 0) yeniMiktar = 0;
                uf.setAmount(yeniMiktar);
                userFridgeRepository.save(uf);
            }
        }
    }
}
