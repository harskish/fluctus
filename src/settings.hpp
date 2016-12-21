#pragma once

#include <string>
#include "json.hpp"

class Settings
{
public:
    // Singleton pattern
    static Settings &getInstance() {
        static Settings instance;
        return instance;
    }
    Settings(Settings const&) = delete;
    void operator=(Settings const&) = delete;

    // Getters
    std::string getPlatformName() { return platformName; }
    std::string getDeviceName() { return deviceName; }
    int getWindowWidth() { return windowWidth; };
    int getWindowHeight() { return windowHeight; };
    float getRenderScale() { return renderScale; };

private:
    Settings();
    void init();
    void load();
    void import(nlohmann::json j);

    // Contents of settings singleton
    std::string platformName;
    std::string deviceName;
    int windowWidth;
    int windowHeight;
    float renderScale;
};
