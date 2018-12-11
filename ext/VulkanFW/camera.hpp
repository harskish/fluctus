/*
* Basic camera class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_access.hpp>

class Camera {
private:
    float fov;
    float znear, zfar;

    void updateViewMatrix() {
        glm::mat4 rotM = glm::mat4(1.0f);
        glm::mat4 transM;

        rotM = glm::rotate(rotM, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
        rotM = glm::rotate(rotM, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

        transM = glm::translate(glm::mat4(1.0f), position);

        if (type == CameraType::firstperson) {
            matrices.view = rotM * transM;
        } else {
            matrices.view = transM * rotM;
        }
        matrices.skyboxView = matrices.view;
        matrices.skyboxView[3] = glm::vec4(0, 0, 0, 1);
    };

public:
    enum CameraType
    {
        lookat,
        firstperson
    };
    CameraType type = CameraType::lookat;

    glm::vec3 rotation = glm::vec3();
    glm::vec3 position = glm::vec3();

    float rotationSpeed = 1.0f;
    float movementSpeed = 1.0f;

    struct {
        glm::mat4 perspective;
        glm::mat4 view;
        glm::mat4 skyboxView;
    } matrices;

    struct {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
        bool forward = false;
        bool back = false;
    } keys;

    bool moving() { return keys.left || keys.right || keys.up || keys.down || keys.forward || keys.back; }

    float getNearClip() { return znear; }

    float getFarClip() { return zfar; }

    void setPerspective(float fov, const vk::Extent2D& size, float znear, float zfar) {
        setPerspective(fov, static_cast<float>(size.width) / static_cast<float>(size.height), znear, zfar);
    }

    void setPerspective(float fov, float aspect, float znear, float zfar) {
        this->fov = fov;
        this->znear = znear;
        this->zfar = zfar;
        matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
    };

    void updateAspectRatio(float aspect) { matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar); }

    void updateAspectRatio(const vk::Extent2D& size) { updateAspectRatio((float)size.width / (float)size.height); }

    void setPosition(const glm::vec3& position) {
        this->position = position;
        updateViewMatrix();
    }

    void setRotation(const glm::vec3& rotation) {
        this->rotation = rotation;
        updateViewMatrix();
    };

    void rotate(const glm::vec3& delta) {
        this->rotation += delta;
        updateViewMatrix();
    }

    void setTranslation(const glm::vec3& translation) {
        this->position = translation;
        updateViewMatrix();
    };

    void dolly(float z) { translate({ 0, 0, z }); }

    glm::vec3 getLeft() {
        return glm::vec3(glm::row(matrices.view, 0));
    }

    glm::vec3 getUp() {
        return glm::vec3(glm::row(matrices.view, 1));
    }

    glm::vec3 getFront() {
        return glm::vec3(glm::row(matrices.view, 2));
    }

    void translate(const glm::vec3& delta) {
        glm::mat3 invR = glm::transpose(glm::mat3(matrices.view));
        glm::vec3 result = invR * glm::vec3(-delta.x, delta.y, delta.z);

        position += result;
        updateViewMatrix();
    }

    void update(float deltaTime) {
        if (type == CameraType::firstperson) {
            if (moving()) {
                glm::vec3 direction;
                if (keys.up ^ keys.down) {
                    direction.y = keys.up ? 1.0f : -1.0f;
                }
                if (keys.left ^ keys.right) {
                    direction.x = keys.left ? -1.0f : 1.0f;
                }
                if (keys.forward ^ keys.back) {
                    direction.z = keys.forward ? 1.0f : -1.0f;
                }
                // Normalize to keep speed constant
                if (glm::length(direction) > 0.0f) {
                    translate(glm::normalize(direction) * deltaTime * movementSpeed);
                }
            }
        }
    };

    // Update camera passing separate axis data (gamepad)
    // Returns true if view or position has been changed
    bool updatePad(glm::vec2 axisLeft, glm::vec2 axisRight, float deltaTime) {
        bool retVal = false;

        if (type == CameraType::firstperson) {
            // Use the common console thumbstick layout
            // Left = view, right = move

            const float deadZone = 0.0015f;
            const float range = 1.0f - deadZone;
            glm::vec3 camFront = getFront();

            float moveSpeed = deltaTime * movementSpeed * 2.0f;
            float rotSpeed = deltaTime * rotationSpeed * 50.0f;

            // Move
            if (fabsf(axisLeft.y) > deadZone) {
                float pos = (fabsf(axisLeft.y) - deadZone) / range;
                position -= camFront * pos * ((axisLeft.y < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
                retVal = true;
            }
            if (fabsf(axisLeft.x) > deadZone) {
                float pos = (fabsf(axisLeft.x) - deadZone) / range;
                position += glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * pos * ((axisLeft.x < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
                retVal = true;
            }

            // Rotate
            if (fabsf(axisRight.x) > deadZone) {
                float pos = (fabsf(axisRight.x) - deadZone) / range;
                rotation.y += pos * ((axisRight.x < 0.0f) ? -1.0f : 1.0f) * rotSpeed;
                retVal = true;
            }
            if (fabsf(axisRight.y) > deadZone) {
                float pos = (fabsf(axisRight.y) - deadZone) / range;
                rotation.x -= pos * ((axisRight.y < 0.0f) ? -1.0f : 1.0f) * rotSpeed;
                retVal = true;
            }
        } else {
            // todo: move code from example base class for look-at
        }

        if (retVal) {
            updateViewMatrix();
        }

        return retVal;
    }
};
