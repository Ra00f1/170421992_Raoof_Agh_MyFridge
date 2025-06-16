package com.example.dbapi.controller;

import com.example.dbapi.dto.ProductAddRequestDto;
import com.example.dbapi.model.UserFridge;
import com.example.dbapi.service.FridgeService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/fridge")
public class FridgeController 
{

    private final FridgeService fridgeService;

    public FridgeController(FridgeService fridgeService) 
    {
        this.fridgeService = fridgeService;
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<List<UserFridge>> getFridgeByUserId(@PathVariable int userId) 
    {
        List<UserFridge> items = fridgeService.getAllProductsForUser(userId);
        return ResponseEntity.ok(items);
    }

    @PostMapping("/add")
    public ResponseEntity<String> addProductToFridge(@RequestBody ProductAddRequestDto dto) 
    {
        try 
        {
            fridgeService.addProductToUserFridge(
                    dto.getUserId(),
                    dto.getCategoryName(),
                    dto.getProductName(),
                    dto.getAmount(),
                    dto.getPrice(),
                    dto.getCountType(),
                    dto.getSkt()
            );
            return ResponseEntity.ok("Ürün başarıyla eklendi.");
        } 
        catch (Exception e)
        {
            return ResponseEntity.badRequest().body("Hata: " + e.getMessage());
        }
    }

    @GetMapping("/user/{userId}/expiring")
    public ResponseEntity<List<UserFridge>> getExpiringProducts(@PathVariable int userId) 
    {
        List<UserFridge> items = fridgeService.getExpiringProductsForUser(userId);
        return ResponseEntity.ok(items);
    }

    @GetMapping("/user/{userId}/expiring-3days")
    public ResponseEntity<List<UserFridge>> getExpiringIn3Days(@PathVariable int userId) 
    {
        List<UserFridge> items = fridgeService.getExpiringSoonIn3Days(userId);
        return ResponseEntity.ok(items);
    }

    @DeleteMapping("/delete")
    public ResponseEntity<String> deleteFromUserFridge(
            @RequestParam Long userId,
            @RequestParam String productName) 
    {
        boolean deleted = fridgeService.deleteProduct(userId, productName);
        if (deleted) 
        {
            return ResponseEntity.ok("Ürün başarıyla silindi.");
        } 
        else 
        {
            return ResponseEntity.badRequest().body("Ürün bulunamadı.");
        }
    }

    @PostMapping("/eksilt")
    public ResponseEntity<String> eksiltProductFromFridge(
            @RequestParam Long userId,
            @RequestParam String productName,
            @RequestParam int miktar) 
            {
        try 
        {
            fridgeService.eksiltProduct(userId, productName.toLowerCase(), miktar);
            return ResponseEntity.ok("Ürün eksiltildi.");
        } 
        catch (Exception e) 
        {
            return ResponseEntity.badRequest().body("Hata: " + e.getMessage());
        }
    }


}
