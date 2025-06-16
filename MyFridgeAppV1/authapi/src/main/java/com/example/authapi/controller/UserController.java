package com.example.authapi.controller;

import com.example.authapi.model.User;
import com.example.authapi.service.UserService;
import com.example.authapi.repository.UserRepository;
import com.example.authapi.model.LoginRequest;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/auth")
public class UserController 
{

    private final UserService userService;
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public UserController(UserService userService, UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userService = userService;
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody User user) 
    {
        userService.register(user);
        return ResponseEntity.ok("Kayıt başarılı");
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) 
    {
        Optional<User> userOpt = userRepository.findByUsername(request.getUsername());

        if (userOpt.isEmpty()) 
        {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Kullanıcı bulunamadı"));
        }

        User user = userOpt.get();
        if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) 
        {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Şifre yanlış"));
        }

        return ResponseEntity.ok(Map.of(
                "message", "Giriş başarılı",
                "userId", user.getId(),
                "username", user.getUsername(),
                "email", user.getEmail(),
                "age", user.getAge(),
                "height", user.getHeight(),
                "weight", user.getWeight(),
                "gender", user.getGender()
        ));
    }
}
