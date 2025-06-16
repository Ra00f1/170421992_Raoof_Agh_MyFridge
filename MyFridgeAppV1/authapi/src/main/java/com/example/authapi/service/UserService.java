package com.example.authapi.service;

import com.example.authapi.model.LoginRequest;
import com.example.authapi.model.User;
import com.example.authapi.repository.UserRepository;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import java.util.Optional;

@Service
public class UserService 
{

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public UserService(UserRepository userRepository,
                       PasswordEncoder passwordEncoder) 
                       {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    public String register(User user) 
    {
        try 
        {
            // Kayıtlama: Kullanıcı adı ve e-posta benzersiz olmalı
            if (userRepository.findByUsername(user.getUsername()).isPresent()) 
            {
                return "Kullanıcı adı zaten alınmış";
            }
            // Şifreyi hashle
            String hashedPassword = passwordEncoder.encode(user.getPassword());
            user.setPassword(hashedPassword);
            user.toString();

            userRepository.save(user);
            return "Kayıt başarılı";
        }
        catch (Exception e)
        {
            return "Kayıt sırasında hata oluştu: " + e.getMessage();
        }
    }

    public ResponseEntity<?> login(LoginRequest request) 
    {
        Optional<User> optionalUser = userRepository.findByUsername(request.getUsername());
        if (optionalUser.isEmpty()) 
        {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Kullanıcı bulunamadı");
        }
        if (optionalUser.isPresent()) 
        {
            User user = optionalUser.get();
            if (passwordEncoder.matches(request.getPassword(), user.getPassword())) 
            {
                return ResponseEntity.ok("Giriş başarılı");
            }
            else
            {
                return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body("Şifre yanlış");
            }
        }

        return ResponseEntity.ok("Giriş başarılı");
    }
}
