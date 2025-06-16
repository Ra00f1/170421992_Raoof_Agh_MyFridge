package com.example.mainapi.controller;

import com.example.mainapi.dto.OcrProductDto;
import com.example.mainapi.service.ProductService;
import com.example.mainapi.service.RecipeService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/products")
public class ProductController 
{

    private final ProductService productService;


    public ProductController(ProductService productService, RecipeService recipeService) 
    {
        this.productService = productService;
    }

    @PostMapping("/upload")
    public ResponseEntity<String> uploadProducts(@RequestBody OcrProductDto productDto) 
    {
        productService.processOcrProducts(productDto);
        return ResponseEntity.ok("OCR verileri başarıyla işlendi.");
    }

    @DeleteMapping("/delete")
    public ResponseEntity<String> deleteProductFromUserFridge(
            @RequestParam Long userId,
            @RequestParam String productName) 
    {
        boolean result = productService.deleteProductFromFridge(userId, productName);
        if (result) 
        {
            return ResponseEntity.ok("Ürün başarıyla silindi.");
        } 
        else 
        {
            return ResponseEntity.badRequest().body("Ürün silinemedi veya bulunamadı.");
        }
    }

}